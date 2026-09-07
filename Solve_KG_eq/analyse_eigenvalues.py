"""
analyse_eigenvalues.py

For each (kappa, N, Nt) dataset in data_dir:
  1. Load all_eigenvalues1  (N complex values, numpy eig order).
     Sort ascending by real part  →  approximates k=1..N ordering.
  2. Load coefficients1  (M x N complex matrix, valid eigenmodes in basis-1).
     Each row is one eigenmode's coefficient vector; find its dominant
     basis index = argmax(|row|) and sort rows small-k → large-k.
  3. Load eigenvalues1  (M real valid eigenvalues).
     Sort in the same dominant-k order as coefficients.
  4. First-knee analysis on the sorted all_eigenvalues1 to find threshold.
     The threshold is the first knee of a 3-segment piecewise-linear fit to
     log10(1 - eigenvalue) vs. index — the transition from the steep initial
     drop to the plateau. Falls back to the top-down asymptote threshold
     (any eigenvalue within 0.1% of the maximum plateau value) when there is
     not enough data to fit three segments.
  5. Print summary table and save plots.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_dir = "data/diff_m/"
out_dir  = "plots/"
os.makedirs(out_dir, exist_ok=True)

THRESHOLD_CAP = 0.99999  # any eigenvalue above this is always valid


# ── File loaders ─────────────────────────────────────────────────────────────

def load_complex_1d(path):
    """Load a 1-column file of complex numbers  (a+bj)  per line."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return np.array([complex(l) for l in lines])


def load_real_1d(path):
    """Load a 1-column file of real floats, one per line."""
    return np.loadtxt(path, dtype=float)


def load_complex_2d(path):
    """
    Load a 2-D file where each row contains space-separated complex numbers
    in numpy's  (real+imagj)  savetxt format.
    Returns complex array of shape  (n_rows, n_cols).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([complex(tok) for tok in line.split()])
    return np.array(rows)           # shape (M, N)

def find_plateau_threshold(ev_asc, tolerance=0.001):
    """
    Top-Down Asymptote Threshold.
    
    Instead of finding a geometric knee, this algorithm anchors to the top 
    of the plateau. It defines "valid solutions" as any mode whose eigenvalue 
    is within a `tolerance` (e.g., 0.001 = 0.1%) of the maximum plateau value.
    """
    ev = np.asarray(ev_asc, dtype=float)
    n = len(ev)
    
    if n == 0 or (ev[-1] - ev[0]) < 1e-12:
        return float(ev[0]), float(ev.mean()), 0.0, 0
    
    # Anchor to the top of the plateau. 
    # We take the median of the top 5% of modes to make it robust against 
    # a single numerical outlier at the very edge.
    top_n = max(1, n // 20)
    plateau_anchor = np.median(ev[-top_n:])
    
    # Threshold is defined strictly as a fixed drop from the plateau anchor.
    # E.g., if anchor is 0.999 and tolerance is 0.001, threshold is 0.998.
    threshold = plateau_anchor * (1 - tolerance)
    
    # Clamp to bounds
    threshold = max(float(ev[0]), min(float(ev[-1]), threshold))

    # The discrete knee index is the first eigenvalue strictly above the threshold
    knee_idx = int(np.searchsorted(ev, threshold))
    if knee_idx >= n:
        knee_idx = n - 1

    plateau = ev[knee_idx:]
    mean_p = float(plateau.mean()) if len(plateau) > 0 else float(ev[-1])
    osc_p = float(plateau.max() - plateau.min()) if len(plateau) > 0 else 0.0
    
    return threshold, mean_p, osc_p, knee_idx


def find_first_knee_piecewise(ev_asc, min_seg=3):
    """
    Three-segment piecewise-linear knee detector on  y = log10(1 - eigenvalue)
    vs. index x.

    Brute-forces both breakpoints (i, j) splitting the curve into
    [0,i) / [i,j) / [j,end), fits an OLS line to each segment, and picks the
    (i, j) that minimizes the total residual sum of squares (the classic
    "L-method" for knee/elbow detection, generalized to two breakpoints).
    The *first* knee — the transition from the steep initial drop to the
    plateau — is then the intersection of segment 1 and segment 2.

    Returns a dict with the fitted segments and the knee location, or None
    if there isn't enough valid data to fit three segments.
    """
    ev = np.asarray(ev_asc, dtype=float)
    diff = 1.0 - ev
    valid = diff > 0
    x = np.arange(len(ev))[valid]
    y = np.log10(diff[valid])
    m = len(x)

    if m < 3 * min_seg:
        return None

    # Prefix sums for O(1) OLS fit + SSR on any segment [a, b).
    cs_x  = np.concatenate(([0.0], np.cumsum(x)))
    cs_y  = np.concatenate(([0.0], np.cumsum(y)))
    cs_xx = np.concatenate(([0.0], np.cumsum(x * x)))
    cs_xy = np.concatenate(([0.0], np.cumsum(x * y)))
    cs_yy = np.concatenate(([0.0], np.cumsum(y * y)))

    def seg_fit(a, b):
        n_s = b - a
        sx, sy   = cs_x[b] - cs_x[a],  cs_y[b] - cs_y[a]
        sxx, sxy = cs_xx[b] - cs_xx[a], cs_xy[b] - cs_xy[a]
        syy      = cs_yy[b] - cs_yy[a]
        denom = n_s * sxx - sx * sx
        if denom == 0:
            slope, intercept = 0.0, sy / n_s
        else:
            slope = (n_s * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n_s
        ssr = (syy - 2 * slope * sxy - 2 * intercept * sy
               + slope**2 * sxx + 2 * slope * intercept * sx
               + intercept**2 * n_s)
        return slope, intercept, max(ssr, 0.0)

    best = None
    for i in range(min_seg, m - 2 * min_seg + 1):
        for j in range(i + min_seg, m - min_seg + 1):
            s1, b1, r1 = seg_fit(0, i)
            s2, b2, r2 = seg_fit(i, j)
            s3, b3, r3 = seg_fit(j, m)
            total = r1 + r2 + r3
            if best is None or total < best[0]:
                best = (total, i, j, (s1, b1), (s2, b2), (s3, b3))

    if best is None:
        return None

    _, i, j, (s1, b1), (s2, b2), (s3, b3) = best
    if np.isclose(s1, s2):
        return None

    x_knee = (b2 - b1) / (s1 - s2)
    y_knee = s1 * x_knee + b1
    ev_knee = 1.0 - 10 ** y_knee

    return {
        "x": x, "y": y,                      # transformed data actually fit
        "i": i, "j": j,                      # breakpoint indices into x/y
        "seg1": (s1, b1), "seg2": (s2, b2), "seg3": (s3, b3),
        "x_knee": float(x_knee), "y_knee": float(y_knee),
        "ev_knee": float(ev_knee),
    }

# ── File discovery ────────────────────────────────────────────────────────────

def collect_datasets(directory):
    """
    Return a list of dicts, one per unique (kappa, N, Nt) that has all six
    required files: all_eigenvalues1, all_eigenvalues2,
                    eigenvalues1,     eigenvalues2,
                    coefficients1,    coefficients2.
    """
    required = {"all_eigenvalues1", "all_eigenvalues2",
                "eigenvalues1",     "eigenvalues2",
                "coefficients1",    "coefficients2"}

    pattern = re.compile(
        r"^(all_eigenvalues1|all_eigenvalues2|eigenvalues1|eigenvalues2"
        r"|coefficients1|coefficients2)"
        r"_2d_N(\d+)_Nt(\d+)_T([\d.]+)_m([\d.]+)\.txt$"
    )

    by_key = {}
    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if not m:
            continue
        ftype, N, Nt, T, kappa = (m.group(1), int(m.group(2)),
                                   int(m.group(3)), float(m.group(4)),
                                   float(m.group(5)))
        key = (kappa, N, Nt, T)
        if key not in by_key:
            by_key[key] = {}
        by_key[key][ftype] = fname

    datasets = []
    for (kappa, N, Nt, T), files in sorted(by_key.items()):
        if required.issubset(files.keys()):
            datasets.append({"kappa": kappa, "N": N, "Nt": Nt, "T": T,
                             **files})
    return datasets


# ── Main analysis ─────────────────────────────────────────────────────────────

datasets = collect_datasets(data_dir)
if not datasets:
    raise RuntimeError(f"No complete datasets found in {data_dir}")
print(f"Found {len(datasets)} dataset(s).\n")

results = []

for ds in datasets:
    kappa = ds["kappa"]
    N     = ds["N"]
    Nt    = ds["Nt"]

    # ── 1. All eigenvalues: sort ascending by real part (≈ ascending k) ──────
    all_ev1 = load_complex_1d(data_dir + ds["all_eigenvalues1"]).real
    all_ev2 = load_complex_1d(data_dir + ds["all_eigenvalues2"]).real
    all_ev1_asc = np.sort(all_ev1)
    all_ev2_asc = np.sort(all_ev2)

    # ── 2. Valid eigenmodes: sort by dominant basis-1 mode (small k → large k)
    coeffs1 = load_complex_2d(data_dir + ds["coefficients1"])  # (M, N)
    coeffs2 = load_complex_2d(data_dir + ds["coefficients2"])  # (M, N)
    valid_ev1 = load_real_1d(data_dir + ds["eigenvalues1"])     # (M,)
    valid_ev2 = load_real_1d(data_dir + ds["eigenvalues2"])     # (M,)

    # dominant k index (0-based) for each valid eigenmode
    dom_k1 = np.argmax(np.abs(coeffs1), axis=1)   # shape (M,)
    dom_k2 = np.argmax(np.abs(coeffs2), axis=1)

    order1 = np.argsort(dom_k1)
    order2 = np.argsort(dom_k2)

    valid_ev1_sorted  = valid_ev1[order1]
    valid_ev2_sorted  = valid_ev2[order2]
    coeffs1_sorted    = coeffs1[order1]
    coeffs2_sorted    = coeffs2[order2]
    dom_k1_sorted     = dom_k1[order1] + 1        # 1-indexed
    dom_k2_sorted     = dom_k2[order2] + 1

    # ── 3. First-knee threshold from all_eigenvalues (sorted ascending) ───────
    # Primary: first knee of a 3-segment piecewise-linear fit to
    # log10(1 - eigenvalue) vs index — the transition from the steep initial
    # drop to the plateau. Falls back to the top-down asymptote threshold
    # when there isn't enough data to fit three segments (knee_fit is None).
    # The threshold is capped at THRESHOLD_CAP: any eigenvalue above that is
    # always valid, even if the knee fit would otherwise place the threshold
    # higher (e.g. for small kappa, where the knee lands near the max value).
    knee_fit1 = find_first_knee_piecewise(all_ev1_asc)
    knee_fit2 = find_first_knee_piecewise(all_ev2_asc)

    if knee_fit1 is not None:
        thr1 = float(np.clip(knee_fit1["ev_knee"], all_ev1_asc[0], all_ev1_asc[-1]))
    else:
        thr1, _, _, _ = find_plateau_threshold(all_ev1_asc)
    thr1 = min(thr1, THRESHOLD_CAP)
    knee1 = min(int(np.searchsorted(all_ev1_asc, thr1)), len(all_ev1_asc) - 1)
    plateau1 = all_ev1_asc[knee1:]
    mean_p1  = float(plateau1.mean())
    osc1     = float(plateau1.max() - plateau1.min()) if len(plateau1) > 1 else 0.0

    if knee_fit2 is not None:
        thr2 = float(np.clip(knee_fit2["ev_knee"], all_ev2_asc[0], all_ev2_asc[-1]))
    else:
        thr2, _, _, _ = find_plateau_threshold(all_ev2_asc)
    thr2 = min(thr2, THRESHOLD_CAP)
    knee2 = min(int(np.searchsorted(all_ev2_asc, thr2)), len(all_ev2_asc) - 1)
    plateau2 = all_ev2_asc[knee2:]
    mean_p2  = float(plateau2.mean())
    osc2     = float(plateau2.max() - plateau2.min()) if len(plateau2) > 1 else 0.0

    n_valid_plateau1 = int(np.sum(all_ev1 >= thr1))
    n_valid_plateau2 = int(np.sum(all_ev2 >= thr2))
    n_valid_old      = len(valid_ev1)

    results.append({
        "kappa": kappa, "N": N, "Nt": Nt,
        "thr1": thr1, "mean_p1": mean_p1, "osc1": osc1, "knee1": knee1,
        "thr2": thr2, "mean_p2": mean_p2, "osc2": osc2, "knee2": knee2,
        "knee_fit1": knee_fit1, "knee_fit2": knee_fit2,
        "n_valid_plateau1": n_valid_plateau1,
        "n_valid_plateau2": n_valid_plateau2,
        "n_valid_old": n_valid_old,
        "all_ev1_asc": all_ev1_asc,
        "all_ev2_asc": all_ev2_asc,
        "valid_ev1_sorted": valid_ev1_sorted,
        "valid_ev2_sorted": valid_ev2_sorted,
        "dom_k1_sorted": dom_k1_sorted,
        "dom_k2_sorted": dom_k2_sorted,
        "coeffs1_sorted": coeffs1_sorted,
        "coeffs2_sorted": coeffs2_sorted,
    })

# ── Print summary table ───────────────────────────────────────────────────────

print(f"{'kappa':>7} {'N':>4} {'Nt':>5} "
      f"{'threshold1':>12} {'mean_plat1':>12} {'osc1':>10} "
      f"{'n_plat1':>8} {'n_plat2':>8} {'n_old':>7} {'frac':>6}")
print("-" * 90)
for r in results:
    print(f"{r['kappa']:7.3f} {r['N']:4d} {r['Nt']:5d} "
          f"{r['thr1']:12.8f} {r['mean_p1']:12.8f} {r['osc1']:10.2e} "
          f"{r['n_valid_plateau1']:8d} {r['n_valid_plateau2']:8d} "
          f"{r['n_valid_old']:7d} {r['n_valid_plateau1']/r['N']:6.3f}")


# ── Helpers for plotting ──────────────────────────────────────────────────────

def plot_spectrum(ax, r, show_dominant_k=False):
    """
    Draw the eigenvalue spectrum for one dataset onto ax.
    If show_dominant_k: use the dominant-mode sorted valid eigenvalues on x.
    Otherwise: use the full ascending all_ev1 with integer index on x.
    """
    if show_dominant_k and len(r["dom_k1_sorted"]) > 0:
        xvals = r["dom_k1_sorted"]
        ev    = r["valid_ev1_sorted"]
        xlabel = "Dominant basis-1 mode k"
    else:
        ev    = r["all_ev1_asc"]
        xvals = np.arange(len(ev))
        xlabel = "Index (sorted ascending)"

    thr = r["thr1"]
    mp  = r["mean_p1"]
    osc = r["osc1"]
    ki  = r["knee1"]  # This is now the index where it crosses the threshold

    vm = ev >= thr
    ax.scatter(xvals[~vm], ev[~vm], color="tomato",    s=18, zorder=3, label="below thr")
    ax.scatter(xvals[vm],  ev[vm],  color="steelblue", s=18, zorder=3, label="valid")
    ax.axhline(thr, color="darkorange", lw=1.4, ls="--",
               label=f"thr={thr:.6f}")
    ax.axhline(mp,  color="green",      lw=1.0, ls=":",
               label=f"mean={mp:.6f}")
    ax.fill_between([xvals.min(), xvals.max()], mp - osc, mp + osc,
                    alpha=0.10, color="green")
    if not show_dominant_k:
        ax.axvline(ki, color="purple", lw=1.0, ls="-.", label=f"thr_idx={ki}")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Eigenvalue", fontsize=8)
    N = r["N"]
    ax.set_title(
        f"κ={r['kappa']:.3f}  N={N}  Nt={r['Nt']}\n"
        f"n_valid={r['n_valid_plateau1']}  ({r['n_valid_plateau1']/N*100:.0f}%)",
        fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(labelsize=7)


# ── Plot 1: summary — threshold and valid fraction vs kappa ──────────────────
# Use only datasets whose Nt = 15*N (the main production series), falling
# back to all datasets if none match (e.g. the series uses a different ratio).

main = [r for r in results if r["Nt"] == 15 * r["N"]]
if not main:
    main = results
if main:
    kappas  = np.array([r["kappa"]           for r in main])
    thrs    = np.array([r["thr1"]            for r in main])
    fracs   = np.array([r["n_valid_plateau1"] / r["N"] for r in main])
    n_valid = np.array([r["n_valid_plateau1"] for r in main])
    Ns      = np.array([r["N"]               for r in main])

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    Nt_ratio = main[0]["Nt"] / main[0]["N"]
    ax1a.plot(kappas, thrs, "o-", color="steelblue", ms=4, lw=1.2)
    ax1a.set_ylabel("First-knee threshold (eigenvalue)", fontsize=9)
    ax1a.set_title(f"First-Knee Threshold vs κ  (Nt = {Nt_ratio:g}N series)", fontsize=10)
    ax1a.grid(True, alpha=0.3)

    ax1b.plot(kappas, fracs,   "s-", color="darkorange", ms=4, lw=1.2, label="n_valid / N")
    ax1b.plot(kappas, n_valid, "^:", color="steelblue",  ms=3, lw=0.8, label="n_valid (abs)")
    ax1b_r = ax1b.twinx()
    ax1b_r.plot(kappas, Ns, "k--", lw=0.8, alpha=0.5, label="N")
    ax1b.set_ylabel("Valid fraction  n_valid / N", fontsize=9)
    ax1b_r.set_ylabel("N  (total modes)", fontsize=9)
    ax1b.set_xlabel("κ", fontsize=9)
    ax1b.set_ylim(0, 1.05)
    lines  = ax1b.get_lines() + ax1b_r.get_lines()
    labels = [l.get_label() for l in lines]
    ax1b.legend(lines, labels, fontsize=7, loc="lower right")
    ax1b.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig(out_dir + "summary_threshold_vs_kappa.png", dpi=150)
    print(f"\nPlot saved: {out_dir}summary_threshold_vs_kappa.png")


# ── Plot 2: eigenvalue spectra for 6 representative kappa values ──────────────
# Pick 6 kappas spread across the main series; show the ascending spectrum.

if main:
    n_panels = min(6, len(main))
    indices  = np.round(np.linspace(0, len(main) - 1, n_panels)).astype(int)
    samples  = [main[i] for i in indices]

    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    for ax, r in zip(axes2.flat, samples):
        plot_spectrum(ax, r, show_dominant_k=False)

    plt.suptitle("Eigenvalue spectra — First-Knee Threshold", fontsize=11)
    plt.tight_layout()
    fig2.savefig(out_dir + "spectra_representative.png", dpi=150)
    print(f"Plot saved: {out_dir}spectra_representative.png")


# ── Plot 3: multi-N consistency at same kappa ─────────────────────────────────
# For each kappa with ≥2 datasets, overlay all N values on one panel.

from collections import defaultdict
by_kappa = defaultdict(list)
for r in results:
    by_kappa[r["kappa"]].append(r)

multi_kappas = sorted(k for k, v in by_kappa.items() if len(v) >= 2)

if multi_kappas:
    n_mk   = len(multi_kappas)
    ncols  = min(3, n_mk)
    nrows  = (n_mk + ncols - 1) // ncols
    fig3, axes3 = plt.subplots(nrows, ncols,
                                figsize=(5 * ncols, 4.5 * nrows),
                                squeeze=False)

    colors_N = ["steelblue", "darkorange", "green", "purple", "brown", "crimson"]

    for idx, kappa in enumerate(multi_kappas):
        ax  = axes3[idx // ncols][idx % ncols]
        grp = sorted(by_kappa[kappa], key=lambda r: r["N"])

        for ci, r in enumerate(grp):
            ev  = r["all_ev1_asc"]
            x   = np.arange(len(ev))
            col = colors_N[ci % len(colors_N)]
            ax.plot(x, ev, "-", color=col, lw=1.2, alpha=0.8,
                    label=f"N={r['N']} Nt={r['Nt']}")
            ax.axhline(r["thr1"], color=col, lw=0.8, ls="--", alpha=0.7)
            ax.axvline(r["knee1"], color=col, lw=0.8, ls=":", alpha=0.7)

        ax.set_xlabel("Index (ascending eigenvalue)", fontsize=8)
        ax.set_ylabel("Eigenvalue", fontsize=8)
        ax.set_title(f"κ={kappa:.3f} — multi-N comparison\n"
                     f"(dashed=threshold, dotted=crossing index)", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # hide unused panels
    for idx in range(n_mk, nrows * ncols):
        axes3[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle("Threshold consistency across N  (same κ)", fontsize=11)
    plt.tight_layout()
    fig3.savefig(out_dir + "multi_N_consistency.png", dpi=150)
    print(f"Plot saved: {out_dir}multi_N_consistency.png")


# ── Plot 4: coefficient heatmaps — up to 9 representative datasets ────────────

coeff_results = [r for r in results if len(r["dom_k1_sorted"]) > 0]
if len(coeff_results) > 9:
    # Spread evenly across the available kappa range
    idx_sel     = np.round(np.linspace(0, len(coeff_results) - 1, 9)).astype(int)
    coeff_results = [coeff_results[i] for i in idx_sel]
if coeff_results:
    n_cr  = len(coeff_results)
    ncols = min(3, n_cr)
    nrows = (n_cr + ncols - 1) // ncols
    fig4, axes4 = plt.subplots(nrows, ncols,
                                figsize=(5 * ncols, 5 * nrows),
                                squeeze=False)

    for idx, r in enumerate(coeff_results):
        ax     = axes4[idx // ncols][idx % ncols]
        coeffs = r["coeffs1_sorted"]
        dom_k  = r["dom_k1_sorted"]
        M, N   = coeffs.shape

        im = ax.imshow(np.abs(coeffs), aspect="auto", origin="lower",
                       extent=[1, N + 1, 0, M], cmap="viridis")
        ax.plot(dom_k, np.arange(M) + 0.5, "r+", ms=5, label="dominant k")

        thr = r["thr1"]
        n_above = int(np.sum(r["valid_ev1_sorted"] >= thr))
        if n_above < M:
            ax.axhline(M - n_above, color="darkorange", lw=1.2, ls="--",
                       label=f"first-knee threshold (n={n_above})")

        plt.colorbar(im, ax=ax, label="|coeff|", shrink=0.8)
        ax.set_xlabel("Basis-1 mode k", fontsize=8)
        ax.set_ylabel("Eigenmode (sorted by dom. k)", fontsize=8)
        ax.set_title(f"κ={r['kappa']:.3f}  N={N}  Nt={r['Nt']}", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    for idx in range(n_cr, nrows * ncols):
        axes4[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle("Coefficient magnitudes in basis-1  (rows = eigenmodes sorted by dominant k)",
                 fontsize=10)
    plt.tight_layout()
    fig4.savefig(out_dir + "coefficient_heatmap.png", dpi=150)
    print(f"Plot saved: {out_dir}coefficient_heatmap.png")


# ── Plot 5: spectrum + gap sequence (shows knee & threshold clearly) ──────────

_pool5 = main if main else results
if _pool5:
    n_panels = min(6, len(_pool5))
    indices  = np.round(np.linspace(0, len(_pool5) - 1, n_panels)).astype(int)
    samples  = [_pool5[i] for i in indices]

    fig5, axes5 = plt.subplots(2, n_panels, figsize=(3.5 * n_panels, 8))
    if n_panels == 1:
        axes5 = axes5.reshape(2, 1)

    for col, r in enumerate(samples):
        ax_spec = axes5[0, col]
        ax_gap  = axes5[1, col]

        ev    = r["all_ev1_asc"]
        thr   = r["thr1"]
        ki    = r["knee1"]
        diffs = np.diff(ev)

        # ── top: eigenvalue spectrum, plotted as log10(1 - eigenvalue) ─────────
        with np.errstate(divide="ignore", invalid="ignore"):
            ev_log  = np.log10(1.0 - ev)
        ev_log[1.0 - ev <= 0] = np.nan
        thr_log = np.log10(1.0 - thr) if (1.0 - thr) > 0 else np.nan

        xvals = np.arange(len(ev))
        vm    = ev >= thr
        ax_spec.scatter(xvals[~vm], ev_log[~vm], color="tomato",    s=12, zorder=3,
                        label="below thr")
        ax_spec.scatter(xvals[vm],  ev_log[vm],  color="steelblue", s=12, zorder=3,
                        label="plateau")
        ax_spec.axhline(thr_log, color="darkorange", lw=1.4, ls="--",
                        label=f"thr={thr:.4f}")
        ax_spec.axvline(ki,  color="purple",     lw=1.2, ls="-.",
                        label=f"threshold idx={ki}")
        ax_spec.set_xlabel("Index (sorted ascending)", fontsize=8)
        ax_spec.set_ylabel(r"$\log_{10}(1 - \mathrm{eigenvalue})$", fontsize=8)
        ax_spec.set_title(
            f"κ={r['kappa']:.3f}  N={r['N']}  Nt={r['Nt']}\n"
            f"n_valid={r['n_valid_plateau1']}  ({r['n_valid_plateau1']/r['N']*100:.0f}%)",
            fontsize=8)
        ax_spec.legend(fontsize=6, loc="lower right")
        ax_spec.tick_params(labelsize=7)

        # ── bottom: consecutive gap sequence ──────────────────────────────────
        gap_x    = np.arange(len(diffs))

        ax_gap.semilogy(gap_x, diffs, "o-", color="steelblue", ms=4, lw=0.8,
                        alpha=0.7, label="gap")
        
        # Show where the threshold falls in the gap sequence
        if ki > 0:
            ax_gap.axvline(ki - 1, color="purple", lw=1.0, ls="-.", alpha=0.6, 
                           label=f"threshold gap idx={ki-1}")
            
        ax_gap.set_xlabel("Gap index  (= eigenvalue index)", fontsize=8)
        ax_gap.set_ylabel("Gap size  (log scale)", fontsize=8)
        ax_gap.set_title("Consecutive gaps", fontsize=8)
        ax_gap.legend(fontsize=6, loc="upper right")
        ax_gap.tick_params(labelsize=7)

    plt.suptitle(
        "Eigenvalue spectrum (top) and gap sequence (bottom)\n"
        "— orange dashed: first-knee threshold  |  purple dash-dot: threshold index",
        fontsize=10)
    plt.tight_layout()
    fig5.savefig(out_dir + "spectra_with_gaps.png", dpi=150)
    plt.close(fig5)
    print(f"Plot saved: {out_dir}spectra_with_gaps.png")


# ── Plot 6: Dominant k of the valid solution at the threshold vs kappa ────────

kappa_vals, dom_k_vals = [], []
for r in results:
    ev  = r["valid_ev1_sorted"]   # eigenvalues of valid modes, sorted by dom_k
    dom = r["dom_k1_sorted"]      # their dominant k (1-indexed), sorted ascending
    thr = r["thr1"]               # eigenvalue at the threshold (= all_ev1_asc[knee1])
    
    if len(ev) < 2:
        continue
        
    # Find the first index where eigenvalue crosses the threshold
    idx = int(np.searchsorted(ev, thr))
    
    # Interpolate exact fractional k for a perfectly smooth curve
    if idx == 0:
        k_val = float(dom[0])
    elif idx >= len(ev):
        k_val = float(dom[-1])
    else:
        ev_below = ev[idx - 1]
        ev_above = ev[idx]
        k_below  = dom[idx - 1]
        k_above  = dom[idx]
        
        # Protect against division by zero just in case eigenvalues are identical
        if ev_above == ev_below:
            k_val = float(k_below)
        else:
            fraction = (thr - ev_below) / (ev_above - ev_below)
            k_val = k_below + fraction * (k_above - k_below)

    kappa_vals.append(r["kappa"])
    dom_k_vals.append(k_val)

if kappa_vals:
    fig6, ax6 = plt.subplots(figsize=(7, 4))
    ax6.plot(kappa_vals, dom_k_vals, "o-", markersize=4)
    # Note: MaxNLocator(integer=True) was removed because k is now fractional/smooth
    ax6.set_xlabel(r"$\kappa$", fontsize=13)
    ax6.set_ylabel(r"Lowest dominant $k$ (interpolated)", fontsize=13)
    ax6.set_title(r"Lowest dominant $k$ vs $\kappa$  (First-Knee Threshold)", fontsize=13)
    ax6.grid(True, linestyle="--", alpha=0.5)
    fig6.tight_layout()
    fig6.savefig(out_dir + "dom_k_vs_kappa.pdf")
    fig6.savefig(out_dir + "dom_k_vs_kappa.png", dpi=150)
    plt.close(fig6)
    print(f"Plot saved: {out_dir}dom_k_vs_kappa.pdf  and  {out_dir}dom_k_vs_kappa.png")