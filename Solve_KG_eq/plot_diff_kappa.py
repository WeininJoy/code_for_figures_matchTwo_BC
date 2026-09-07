import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

data_dir    = "data/diff_m/"
figures_dir = "figures/diff_m/"
os.makedirs(figures_dir, exist_ok=True)
T = np.pi
L = 2.0 * np.pi
kappa_list = np.linspace(0.01, 1, num=100)
N_plot    = 4
N_t_plot  = 300
N_x_plot  = 300

THRESHOLD_CAP = 0.99999  # any eigenvalue above this is always valid


def get_N_Nt(kappa):
    # Must match the formula in solve_pde_diff_kappa.py _worker()
    N  = max(30, int(kappa * 150))
    Nt = 5 * N
    return N, Nt



def find_plateau_threshold(ev_asc, tolerance=0.001):
    """
    Top-Down Asymptote Threshold.

    Instead of finding a geometric knee, this algorithm anchors to the top
    of the plateau. It defines "valid solutions" as any mode whose eigenvalue
    is within a `tolerance` (e.g., 0.001 = 0.1%) of the maximum plateau value.

    Used only as a fallback for find_first_knee_piecewise() when there isn't
    enough data to fit three segments.
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


def find_threshold(ev_asc):
    """
    Primary threshold: first knee of a 3-segment piecewise-linear fit
    (matches analyse_eigenvalues.py). Falls back to the top-down asymptote
    threshold when there isn't enough data to fit three segments. Capped at
    THRESHOLD_CAP: any eigenvalue above that is always valid, even if the
    knee fit would otherwise place the threshold higher.
    """
    ev_asc = np.asarray(ev_asc, dtype=float)
    knee_fit = find_first_knee_piecewise(ev_asc)
    if knee_fit is not None:
        threshold = float(np.clip(knee_fit["ev_knee"], ev_asc[0], ev_asc[-1]))
    else:
        threshold, _, _, _ = find_plateau_threshold(ev_asc)
    return min(threshold, THRESHOLD_CAP)


def _load_real_1d(path):
    """Load a 1-D eigenvalue file, handling both plain-float and (a+bj) formats."""
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    try:
        return np.array([float(l) for l in lines])
    except ValueError:
        return np.array([complex(l).real for l in lines])


def load_data(kappa):
    N, Nt = get_N_Nt(kappa)
    tag   = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.3f}"

    # Compute the physical knee threshold from all N eigenvalues
    all_ev    = _load_real_1d(f"{data_dir}all_eigenvalues1_2d_{tag}.txt")
    threshold = find_threshold(np.sort(all_ev))

    # Load eigenmodes pre-filtered by the solve script (|ev - 1| < 0.01)
    eig1  = _load_real_1d(f"{data_dir}eigenvalues1_2d_{tag}.txt")
    coef1 = np.loadtxt(f"{data_dir}coefficients1_2d_{tag}.txt", dtype=np.complex128)
    eig2  = _load_real_1d(f"{data_dir}eigenvalues2_2d_{tag}.txt")
    coef2 = np.loadtxt(f"{data_dir}coefficients2_2d_{tag}.txt", dtype=np.complex128)

    if coef1.ndim == 1: coef1 = coef1[np.newaxis, :]
    if coef2.ndim == 1: coef2 = coef2[np.newaxis, :]

    # Compute dom_k sorted arrays on the FULL set (matches analyse_eigenvalues.py)
    dom_k_raw  = np.argmax(np.abs(coef1), axis=1)
    order_raw  = np.argsort(dom_k_raw)
    eig1_by_dk = eig1[order_raw]          # eigenvalues sorted by dom_k
    dom_by_dk  = dom_k_raw[order_raw]     # dom_k indices (0-based), sorted

    # Apply the physical knee threshold (more selective than the solve-script filter)
    mask  = eig1 >= threshold
    eig1, coef1 = eig1[mask],  coef1[mask]
    eig2, coef2 = eig2[mask],  coef2[mask]

    print(f"kappa={kappa:.3f}: threshold={threshold:.7f}  "
          f"n_valid={mask.sum()} / {len(mask)}", flush=True)

    return N, Nt, eig1, coef1, eig2, coef2, threshold, eig1_by_dk, dom_by_dk


def build_solutions(kappa, coef1, coef2, N):
    t_list = np.linspace(0, T, N_t_plot)
    x_list = np.linspace(0, L, N_x_plot)
    t_grid, x_grid = np.meshgrid(t_list, x_list)

    basis_1 = [np.cos(np.sqrt(k**2 + kappa**2) * t_grid) * np.exp(1j * k * x_grid)
               for k in range(1, N + 1)]

    if (np.pi / T) ** 2 - kappa**2 < 0:
        start_n = round(np.sqrt(abs((np.pi / T) ** 2 - kappa**2))) + 1
    else:
        start_n = 1
    basis_2 = [np.cos(n * np.pi / T * t_grid) * np.exp(1j * np.sqrt((n * np.pi / T) ** 2 - kappa**2) * x_grid)
               for n in range(start_n, N + start_n)]

    n_sol = coef1.shape[0]
    sol1_list, sol2_list = [], []
    for i in range(n_sol):
        sol1 = sum(coef1[i, j] * basis_1[j] for j in range(N))
        sol2 = sum(coef2[i, j] * basis_2[j] for j in range(N))
        sol1_list.append(sol1)
        sol2_list.append(sol2)

    diff_array = np.zeros((n_sol, n_sol))
    for i in range(n_sol):
        for j in range(n_sol):
            norm1 = np.linalg.norm(sol1_list[i])
            diff_array[i, j] = np.linalg.norm(sol1_list[i] - sol2_list[j]) / (norm1 + 1e-30)
    small_diff_idx = [int(np.argmin(diff_array[i])) for i in range(n_sol)]

    max_indices  = np.argmax(np.abs(coef1), axis=1)
    sorted_index = np.argsort(max_indices).tolist()

    return t_grid, x_grid, t_list, x_list, sol1_list, sol2_list, small_diff_idx, sorted_index, max_indices


# # ── Figure 1: Phi(t, x=L/2) overlay for m=0.1~1.0 ───────────────────────────

# kappa_list_fig1 = kappa_list[kappa_list <= 1]
# fig, ax = plt.subplots(figsize=(8, 5))
# cmap_v = plt.get_cmap("viridis")
# colors = cmap_v(np.linspace(0, 1, len(kappa_list_fig1)))

# for idx, kappa in enumerate(kappa_list_fig1):
#     N, Nt, eig1, coef1, eig2, coef2, threshold = load_data(kappa)
#     if len(coef1) == 0:
#         print(f"kappa={kappa:.3f}: no modes above threshold={threshold:.7f}, skipping")
#         continue

#     dom_k_idx  = np.argmax(np.abs(coef1.real), axis=1)
#     chosen     = np.argmin(dom_k_idx)
#     chosen_c   = coef1[chosen]
#     chosen_dom = dom_k_idx[chosen] + 1

#     t_list  = np.linspace(0, T, N_t_plot)
#     x_half  = L / 2.0
#     phi_t   = np.zeros(N_t_plot, dtype=complex)
#     for j in range(N):
#         k     = j + 1
#         omega = np.sqrt(k**2 + kappa**2)
#         phi_t += chosen_c[j] * np.cos(omega * t_list) * np.exp(1j * k * x_half)

#     ax.plot(t_list, phi_t.real, color=colors[idx],
#             label=rf"$\mu={kappa:.3f}$, dom $k={chosen_dom}$")

# ax.set_xlabel(r"$t$", fontsize=12)
# ax.set_ylabel(r"$\Phi(t,\, x=L/2)$", fontsize=12)
# ax.set_title(r"Solutions $\Phi(t,\,x=L/2)$ for each $\mu$ (physical knee threshold, lowest dominant $k$)")
# ax.legend(fontsize=7, ncol=2, loc="upper right")
# fig.tight_layout()
# plt.savefig(f"{figures_dir}phi_t_diff_m.pdf")
# plt.savefig(f"{figures_dir}phi_t_diff_m.png", dpi=150)
# print(f"Saved {figures_dir}phi_t_diff_m.pdf and {figures_dir}phi_t_diff_m.png")
# plt.close(fig)


# # ── Figure 2: 2×2 contour for selected kappas ────────────────────────────────

# kappa_contour = [0.1, 0.4, 0.7, 1.0]
# cmap_c  = plt.get_cmap("RdGy")
# fig2, axs2 = plt.subplots(2, 2, figsize=(9, 5.5))

# for ax_c, kappa in zip(axs2.flat, kappa_contour):
#     N, Nt, eig1, coef1, eig2, coef2, threshold = load_data(kappa)
#     dom_k_idx = np.argmax(np.abs(coef1.real), axis=1)
#     chosen    = np.argmin(dom_k_idx)
#     chosen_c  = coef1[chosen]
#     chosen_dom = dom_k_idx[chosen] + 1

#     t_list   = np.linspace(0, T, N_t_plot)
#     x_list   = np.linspace(0, L, N_x_plot)
#     t_g, x_g = np.meshgrid(t_list, x_list)

#     solution = np.zeros_like(t_g, dtype=complex)
#     for j in range(N):
#         k     = j + 1
#         omega = np.sqrt(k**2 + kappa**2)
#         solution += chosen_c[j] * np.cos(omega * t_g) * np.exp(1j * k * x_g)

#     cf = ax_c.contourf(x_g, t_g, solution.real, 20, cmap=cmap_c)
#     fig2.colorbar(cf, ax=ax_c)
#     ax_c.set_title(rf"$\mu={kappa:.3f}$, dom $k={chosen_dom}$", fontsize=10)
#     ax_c.set_xlabel(r"$x$", fontsize=9)
#     ax_c.set_ylabel(r"$t$", fontsize=9)

# fig2.suptitle(r"$\Phi(t,x)$ for selected $\mu$ (physical knee threshold, lowest dominant $k$)", fontsize=11)
# fig2.tight_layout()
# plt.savefig(f"{figures_dir}phi_contour_diff_m.pdf")
# plt.savefig(f"{figures_dir}phi_contour_diff_m.png", dpi=150)
# print(f"Saved {figures_dir}phi_contour_diff_m.pdf and {figures_dir}phi_contour_diff_m.png")
# plt.close(fig2)


# ── Figure 3: Lowest dominant k vs mu ────────────────────────────────────────

all_data = {}
for kappa in kappa_list:
    try:
        all_data[kappa] = load_data(kappa)
    except Exception:
        print(f"kappa={kappa:.3f}: data not found, skipping")

fig3, ax3 = plt.subplots(figsize=(6, 4))
kappa_vals, dom_k_vals, thr_vals = [], [], []
for kappa, data in all_data.items():
    N, Nt, eig1, coef1, eig2, coef2, threshold, eig1_by_dk, dom_by_dk = data
    if len(coef1) == 0:
        continue
    # search in full (unfiltered) dom_k-sorted arrays — matches analyse_eigenvalues.py
    idx_closest = int(np.argmin(np.abs(eig1_by_dk - threshold)))
    kappa_vals.append(kappa)
    dom_k_vals.append(int(dom_by_dk[idx_closest]) + 1)
    thr_vals.append(threshold)
ax3.plot(kappa_vals, dom_k_vals, "o-", markersize=4)
ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
ax3.set_xlabel(r"$\mu$", fontsize=13)
ax3.set_ylabel(r"Lowest dominant $k$", fontsize=13)
ax3.set_title(r"Lowest dominant $k$ vs $\mu$ ", fontsize=13)
ax3.grid(True, linestyle="--", alpha=0.5)
fig3.tight_layout()
plt.savefig(f"{figures_dir}dom_k_vs_mu.pdf")
plt.savefig(f"{figures_dir}dom_k_vs_mu.png", dpi=150)
print(f"Saved {figures_dir}dom_k_vs_mu.pdf and {figures_dir}dom_k_vs_mu.png")
plt.close(fig3)


# # ── Per-kappa figures (all figure types from plot_solution_wave_eq.py) ────────

# for kappa in kappa_list[kappa_list <= 1]:
for kappa in np.array([0.1]):

    N, Nt, eig1, coef1, eig2, coef2, threshold, _, _ = load_data(kappa)
    tag    = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.3f}"
    n_sol  = coef1.shape[0]
    n_plot = min(N_plot, n_sol)

    if n_sol == 0:
        print(f"kappa={kappa:.3f}: no valid modes above threshold={threshold:.7f}, skipping")
        continue

    t_grid, x_grid, t_list, x_list, sol1_list, sol2_list, small_diff_idx, sorted_idx, max_indices = \
        build_solutions(kappa, coef1, coef2, N)

    cmap_rdgy = plt.get_cmap("RdGy")

    # Figure A: Coefficients bar plot
    coef_threshold = 0.05
    k_max = 1
    for i in range(n_plot):
        coefs_norm = coef1[sorted_idx[i]].real
        coefs_norm = coefs_norm / (np.max(np.abs(coefs_norm)) + 1e-30)
        hits = np.where(np.abs(coefs_norm) > coef_threshold)[0]
        if len(hits) > 0:
            k_max = max(k_max, hits[-1] + 1)

    fig, axs = plt.subplots(n_plot, figsize=(3.8, 0.7 * n_plot))
    if n_plot == 1:
        axs = [axs]
    fig.suptitle(rf"Coefficients for linear combination, $\mu={kappa:.3f}$ ", fontsize=10)
    for i in range(n_plot):
        idx   = sorted_idx[i]
        coefs = coef1[idx].real
        coefs = coefs / (np.max(np.abs(coefs)) + 1e-30)
        axs[i].bar(range(1, N + 1), coefs.tolist(), width=0.2)
        axs[i].set_xlim(0, k_max + 1)
        axs[i].set_ylim(-1, 1.0)
        axs[i].xaxis.set_tick_params(labelsize=8)
        axs[i].yaxis.set_tick_params(labelsize=8)
        axs[i].label_outer()
    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[-1].set_xlabel(r"$k$", fontsize=9)
    fig.tight_layout()
    plt.savefig(f"{figures_dir}eigenvalue_2d_{tag}_coefficients.pdf")
    plt.close(fig)

    # Figure B: Phi(t, x=L/2) – basis_1 vs basis_2
    fig, axs = plt.subplots(n_plot, figsize=(6, 0.8 * n_plot))
    if n_plot == 1:
        axs = [axs]
    fig.suptitle(rf"$\Phi(t,\,x=L/2)$, $\mu={kappa:.3f}$")
    for i in range(n_plot):
        idx = sorted_idx[i]
        s1  = sol1_list[idx]
        s2  = sol2_list[small_diff_idx[idx]]
        axs[i].plot(t_list, s1[N_x_plot // 2, :].real, color="r", label=r"basis: $\phi$")
        axs[i].plot(t_list, s2[N_x_plot // 2, :].real, color="g", linestyle="dashed",
                    label=r"basis: $\tilde\phi$")
        axs[i].label_outer()
    axs[-1].legend(fontsize=10)
    axs[-1].set_xlabel(r"$t$")
    fig.tight_layout()
    plt.savefig(f"{figures_dir}eigenvalue_2d_{tag}_xpi.pdf")
    plt.close(fig)

    # Figure C: Phi(x, t=T/2) – basis_1 vs basis_2
    fig, axs = plt.subplots(n_plot, figsize=(6, 0.8 * n_plot))
    if n_plot == 1:
        axs = [axs]
    fig.suptitle(rf"$\Phi(x,\,t=T/2)$, $\mu={kappa:.3f}$")
    for i in range(n_plot):
        idx = sorted_idx[i]
        s1  = sol1_list[idx]
        s2  = sol2_list[small_diff_idx[idx]]
        axs[i].plot(x_list, s1[:, N_t_plot // 2].real, color="r", label=r"basis: $\phi$")
        axs[i].plot(x_list, s2[:, N_t_plot // 2].real, color="g", linestyle="dashed",
                    label=r"basis: $\tilde\phi$")
        axs[i].label_outer()
    axs[-1].legend(fontsize=10)
    axs[-1].set_xlabel(r"$x$")
    fig.tight_layout()
    plt.savefig(f"{figures_dir}eigenvalue_2d_{tag}_t0.5pi.pdf")
    plt.close(fig)

    # Figure D: Contour sequence (n_plot//2 rows × 2 cols)
    n_rows = n_plot // 2
    if n_rows > 0:
        fig, axs2d = plt.subplots(n_rows, 2, figsize=(7, 5))
        if n_rows == 1:
            axs2d = axs2d[np.newaxis, :]
        for i in range(n_rows):
            for j in range(2):
                idx      = sorted_idx[2 * i + j]
                solution = sol1_list[idx].real
                axs2d[i, j].contourf(x_grid, t_grid, solution, 20, cmap=cmap_rdgy)
                axs2d[i, j].set_title(f"dom k: {max_indices[idx] + 1:d}", size=10)
        for ax in axs2d.flat:
            ax.set_xlabel(r"$x$", fontsize=9)
            ax.set_ylabel(r"$t$", fontsize=9)
            ax.label_outer()
        fig.tight_layout()
        plt.savefig(f"{figures_dir}eigenvalue_2d_{tag}_contour_sequences.pdf")
        plt.close(fig)

    # Figure E: Single contour (lowest-dominant-k solution)
    idx      = sorted_idx[0]
    solution = sol1_list[idx].real
    fig = plt.figure(figsize=(4.5, 3))
    plt.contourf(x_grid, t_grid, solution, 20, cmap=cmap_rdgy)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.colorbar()
    plt.title(rf"$\Phi(t,x)$, $\mu={kappa:.3f}$, dom $k={max_indices[idx] + 1}$", fontsize=10)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{figures_dir}eigenvalue_2d_{tag}_contour1.pdf")
    plt.close(fig)

    # Figure F: Cylinder plot – prefer solution with k_dom=5, else 6, else 4
    k_dom_all = max_indices + 1
    idx_cyl = None
    for target_k in [5, 6, 4]:
        hits = np.where(k_dom_all == target_k)[0]
        if len(hits) > 0:
            idx_cyl = hits[0]
            break
    if idx_cyl is None:
        idx_cyl = sorted_idx[0]
    solution_cyl = sol1_list[idx_cyl].real

    y_cyl = np.cos(x_grid)
    z_cyl = np.sin(x_grid)
    vmin, vmax = solution_cyl.min(), solution_cyl.max()
    facecolors = cmap_rdgy((solution_cyl - vmin) / (vmax - vmin + 1e-30))
    fig = plt.figure(figsize=(3.2, 3.2))
    ax3d = fig.add_subplot(1, 1, 1, projection="3d")
    ax3d.set_axis_off()
    ax3d.plot_surface(y_cyl, z_cyl, t_grid, rstride=1, cstride=1,
                      facecolors=facecolors, linewidth=0, antialiased=False, alpha=0.9)
    fig.tight_layout()
    plt.savefig(f"{figures_dir}eigenvalue_2d_{tag}_cylinder_kdom{k_dom_all[idx_cyl]}.pdf")
    plt.close(fig)

    print(rf"$\mu={kappa:.3f}$: saved 6 figures to {figures_dir}")
