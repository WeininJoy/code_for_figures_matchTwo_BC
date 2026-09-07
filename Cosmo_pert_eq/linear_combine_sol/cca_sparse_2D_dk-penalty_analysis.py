# -*- coding: utf-8 -*-
"""
CCA-Based Sparse Joint Eigenfunction Method  —  2D Space-Time Variant
with k-Proportional Diagonal Penalty (Continuous Regularisation)
======================================================================

Extension of cca_sparse_2D_reg_analysis.py.  Replaces the hard
identity/rCCA partition with a single continuous k-proportional
regularised CCA applied to all modes simultaneously.

1. k-Proportional Diagonal Penalty  (Step 4)
   Instead of splitting modes into I_degen / I_mixed, we regularise the
   Gram matrices with a diagonal penalty proportional to k:

       G1_reg = G1_2D + alpha * diag(k_values_1)
       G2_reg = G2_2D + alpha * diag(k_values_2)

   At low k the penalty is negligible — physical mixing is preserved.
   At high k the monotonically increasing penalty breaks SVD degeneracy
   without a hard threshold, yielding a smooth transition.

   See smooth_dk_penalty_implementation.md for the full derivation.

2. Sparse rotation
   Promax oblique rotation applied to the full set of valid CCA modes.

Outputs go to ./figures_2D_dk_penalty/
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from multi_perturbation_analysis import (
    generate_multi_perturbation_bases,
    fcb_time,
    OmegaLambda,
    OmegaK,
)
from cca_sparse_analysis import (
    matrix_sqrt_inv,
    solve_cca,
    apply_joint_sparse_rotation,
    gini,
    normalize_columns,
    plot_gram_spectrum,
    plot_canonical_correlations,
    plot_coefficients,
    plot_heatmap,
    plot_reconstructed_timeseries,
    PERTURBATION_TYPES,
)

# ---------------------------------------------------------------------------
OUTPUT_DIR    = f'./figures/'
RESULTS_CACHE = f'cca_2D_dk_penalty_results.pickle'
FOLDER_PATH = f'./data/'
# ---------------------------------------------------------------------------


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ============================================================================
# STEP 1 — Load bases
# ============================================================================

def load_bases(folder_path=FOLDER_PATH, N_t=1000):
    """Load integerK (basis 1) and allowedK (basis 2) onto a uniform eta grid."""
    eta_grid = np.linspace(0, fcb_time, N_t)
    print("Loading integerK (basis 1)...")
    basis_1 = generate_multi_perturbation_bases("integerK", eta_grid,
                                                folder_path=folder_path)
    print("\nLoading allowedK (basis 2)...")
    basis_2 = generate_multi_perturbation_bases("allowedK", eta_grid,
                                                folder_path=folder_path)
    return basis_1, basis_2, eta_grid


# ============================================================================
# STEP 1b — Load k-values and convert to beta
# ============================================================================

def load_kvalues(dataset_name, folder_path=FOLDER_PATH):
    """Load k-value array for a given dataset from the timeseries data directory."""
    path = os.path.join(folder_path,
                        f'data_{dataset_name}_timeseries',
                        'L70_kvalues.npy')
    k_values = np.load(path)
    print(f"  Loaded {len(k_values)} k-values for {dataset_name}  "
          f"(k_min={k_values.min():.4f})")
    return k_values


def k_to_beta(k_values):
    """
    Convert stored k-values to spatial wavenumber beta.

    In a closed universe K=+1, the relation between the dimensionless
    wavenumber beta and the comoving wavenumber k is:

        k^2 = beta^2 - 1   =>   beta = sqrt(k^2 + 1)

    For integerK: beta should be close to integers (3, 4, 5, ...).
    For allowedK: beta takes non-integer values near the same integers.
    """
    return np.sqrt(k_values**2 + 1.0)


# ============================================================================
# STEP 1 — Frobenius normalise
# ============================================================================

def frobenius_normalize(basis_dict):
    """
    Divide each perturbation-type matrix by its Frobenius norm.

    Returns
    -------
    Xn    : dict  {p: normalised (N_t, N_k) array}
    norms : dict  {p: Frobenius norm}
    """
    Xn, norms = {}, {}
    for p in PERTURBATION_TYPES:
        X = basis_dict[p]
        n = np.linalg.norm(X, 'fro')
        norms[p] = n
        Xn[p] = X / n
    return Xn, norms


# ============================================================================
# STEP 2a — Time-only Gram matrices
# ============================================================================

def compute_time_gram_matrices(Xn1, Xn2, weights=None):
    """
    Compute the three time-only k-space Gram matrices:

        G1_time  = sum_p  w_p  (Xn1^p)^T  Xn1^p    shape (N_k1, N_k1)
        G2_time  = sum_p  w_p  (Xn2^p)^T  Xn2^p    shape (N_k2, N_k2)
        G12_time = sum_p  w_p  (Xn1^p)^T  Xn2^p    shape (N_k1, N_k2)
    """
    if weights is None:
        weights = {p: 1.0 for p in PERTURBATION_TYPES}

    N_k1 = Xn1[PERTURBATION_TYPES[0]].shape[1]
    N_k2 = Xn2[PERTURBATION_TYPES[0]].shape[1]
    G1_t  = np.zeros((N_k1, N_k1))
    G2_t  = np.zeros((N_k2, N_k2))
    G12_t = np.zeros((N_k1, N_k2))

    for p in PERTURBATION_TYPES:
        w   = weights[p]
        X1p = Xn1[p]
        X2p = Xn2[p]
        G1_t  += w * X1p.T @ X1p
        G2_t  += w * X2p.T @ X2p
        G12_t += w * X1p.T @ X2p

    return G1_t, G2_t, G12_t


# ============================================================================
# STEP 2b — Spatial overlap matrix
# ============================================================================

def compute_spatial_overlap(beta1, beta2):
    """
    Compute the spatial overlap matrix S[i,j] between modes with wavenumbers
    beta1[i] and beta2[j] in a closed universe (K=+1).

        S[i,j] = (1 / beta1[i] * beta2[j])
                 * (sin(diff*pi)/(2*diff) - sin(summ*pi)/(2*summ))

    The limit diff->0 is handled analytically (limit = pi/2).
    """
    beta1 = np.asarray(beta1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    b1 = beta1[:, None]
    b2 = beta2[None, :]

    diff = b1 - b2
    summ = b1 + b2

    near_zero = np.abs(diff) < 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(near_zero,
                         np.pi / 2.0,
                         np.sin(diff * np.pi) / (2.0 * diff))

    term2 = np.sin(summ * np.pi) / (2.0 * summ)
    S = (term1 - term2) / (b1 * b2)
    return S


# ============================================================================
# STEP 2c — 2D (space-time) Gram matrices via Hadamard product
# ============================================================================

def compute_2d_gram_matrices(G1_t, G2_t, G12_t, S11, S22, S12,
                             normalise_spatial=True):
    """
    Form 2D space-time Gram matrices as Hadamard products:

        G1_2D  = G1_time  * S11
        G2_2D  = G2_time  * S22
        G12_2D = G12_time * S12

    If normalise_spatial=True, S is normalised so diag(S11)=diag(S22)=1,
    preserving the diagonal scale of the time-only Gram matrices.
    """
    if normalise_spatial:
        d11 = np.sqrt(np.diag(S11))
        d22 = np.sqrt(np.diag(S22))
        d11 = np.where(d11 < 1e-30, 1.0, d11)
        d22 = np.where(d22 < 1e-30, 1.0, d22)
        S11_n = S11 / np.outer(d11, d11)
        S22_n = S22 / np.outer(d22, d22)
        S12_n = S12 / np.outer(d11, d22)
    else:
        S11_n, S22_n, S12_n = S11, S22, S12

    G1_2D  = G1_t  * S11_n
    G2_2D  = G2_t  * S22_n
    G12_2D = G12_t * S12_n
    return G1_2D, G2_2D, G12_2D


# ============================================================================
# k-PROPORTIONAL REGULARISED CCA  (Step 3)
# ============================================================================

def solve_rcca_k_scaled(G1, G2, G12, k_array_1, k_array_2,
                        alpha=1e-6, ev_threshold=1e-10, rho_min=0.99):
    """
    Regularised CCA using a k-proportional diagonal penalty.

        G1_reg = G1 + alpha * diag(k_array_1)
        G2_reg = G2 + alpha * diag(k_array_2)

    Because k grows monotonically, the penalty is negligible at low k
    (preserving physical mixing/wiggles) and naturally strong at high k
    (breaking SVD eigenspace degeneracy without a hard partition).

    See smooth_dk_penalty_implementation.md for the full derivation.

    Parameters
    ----------
    G1, G2, G12   : 2D space-time Gram matrices  (N_k, N_k)
    k_array_1     : (N_k,) k-values for basis 1 (integerK)
    k_array_2     : (N_k,) k-values for basis 2 (allowedK)
    alpha         : overall penalty strength (e.g. 1e-5)
    ev_threshold  : relative eigenvalue cutoff for G^{-1/2}
    rho_min       : keep modes with rho_i > rho_min

    Returns
    -------
    A, B, rho, rho_all  (same signature as solve_cca)
    """
    G1_reg = G1 + alpha * np.diag(k_array_1)
    G2_reg = G2 + alpha * np.diag(k_array_2)
    return solve_cca(G1_reg, G2_reg, G12, ev_threshold, rho_min)


# ============================================================================
# Diagnostics
# ============================================================================

def print_beta_info(beta1, beta2):
    """Print beta values and how close they are to integers."""
    print(f"\n  integerK: {len(beta1)} modes,  "
          f"beta min={beta1.min():.4f}, max={beta1.max():.4f}")
    nearest_int = np.round(beta1).astype(int)
    max_err = np.max(np.abs(beta1 - nearest_int))
    print(f"  integerK max deviation from integer: {max_err:.2e}")

    print(f"\n  allowedK: {len(beta2)} modes,  "
          f"beta min={beta2.min():.4f}, max={beta2.max():.4f}")
    nearest_int2 = np.round(beta2).astype(int)
    max_err2 = np.max(np.abs(beta2 - nearest_int2))
    print(f"  allowedK max deviation from integer: {max_err2:.2e}")


# ============================================================================
# Plots
# ============================================================================

def plot_spatial_overlap(S12, beta1, beta2, output_dir=OUTPUT_DIR):
    """Heatmap of the spatial overlap matrix S12 between the two bases."""
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im0 = axes[0].imshow(np.abs(S12), aspect='auto', cmap='viridis',
                         interpolation='nearest')
    axes[0].set_xlabel('allowedK index (beta2)')
    axes[0].set_ylabel('integerK index (beta1)')
    axes[0].set_title('|S12| — raw spatial overlap')
    plt.colorbar(im0, ax=axes[0])

    col_max = np.max(np.abs(S12), axis=0)
    col_max = np.where(col_max < 1e-30, 1.0, col_max)
    S12_n = np.abs(S12) / col_max[None, :]
    im1 = axes[1].imshow(S12_n, aspect='auto', cmap='viridis',
                         interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_xlabel('allowedK index (beta2)')
    axes[1].set_ylabel('integerK index (beta1)')
    axes[1].set_title('|S12| column-normalised')
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(f'Spatial overlap S12  (N1={len(beta1)}, N2={len(beta2)})',
                 fontsize=12)
    fig.tight_layout()
    out = output_dir + '2D_spatial_overlap.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_integer_heatmap_with_k_difference(
        A_sparse, k_values_integer, k_values_allowed,
        output_dir=OUTPUT_DIR):
    """Align the integerK sparse heatmap with allowedK - integerK below it."""
    _ensure_dir(output_dir)
    if len(k_values_integer) != len(k_values_allowed):
        raise ValueError(
            "The combined index plot requires equal integerK and allowedK "
            f"lengths; got {len(k_values_integer)} and {len(k_values_allowed)}"
        )
    if A_sparse.shape[0] != len(k_values_integer):
        raise ValueError(
            f"A_sparse has {A_sparse.shape[0]} k rows but the k arrays have "
            f"length {len(k_values_integer)}"
        )

    A_norm = normalize_columns(A_sparse)
    dominant = np.argmax(np.abs(A_norm), axis=0)
    order = np.argsort(dominant)
    heatmap = np.abs(A_norm[:, order].T)
    H0 = 1.0 / np.sqrt(3.0 * OmegaLambda)
    a0 = 1.0
    K = -OmegaK * a0**2 * H0**2
    curvature_scale = np.sqrt(np.abs(K))
    if curvature_scale < 1e-30:
        raise ValueError("Cannot express k in curvature units when K is zero")
    delta_k = (
        k_values_allowed / curvature_scale
        - k_values_integer / curvature_scale
    )
    index = np.arange(len(delta_k))

    fig = plt.figure(figsize=(6, 6))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[3.2, 1.2], width_ratios=[1.0, 0.035],
        hspace=0.08, wspace=0.04,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[1, 0], sharex=ax_heat)
    ax_colorbar = fig.add_subplot(grid[0, 1])
    # Reserve exactly the same right-hand width on the lower row. This keeps
    # the physical left/right edges of the two data axes perfectly aligned.
    ax_spacer = fig.add_subplot(grid[1, 1])
    ax_spacer.set_axis_off()
    image = ax_heat.imshow(
        heatmap, aspect='auto', cmap='Blues', interpolation='nearest',
        vmin=0, vmax=1, origin='upper',
        extent=(-0.5, len(delta_k) - 0.5, heatmap.shape[0] - 0.5, -0.5),
    )
    ax_heat.set_ylabel('CCA mode\n(sorted by dominant integerK index)')
    ax_heat.set_title(
        'IntegerK sparse coefficients and matched wavenumber difference')
    colorbar = fig.colorbar(image, cax=ax_colorbar)
    colorbar.set_label(r'$|A_{ij}|$ (column-normalised)')
    ax_heat.tick_params(axis='x', labelbottom=False)

    ax_delta.axhline(0.0, color='black', lw=0.8)
    ax_delta.plot(index, delta_k, color='tab:red', lw=1.2)
    ax_delta.fill_between(index, 0.0, delta_k, color='tab:red', alpha=0.18)
    ax_delta.set_xlim(-0.5, len(delta_k) - 0.5)
    ax_delta.set_xlabel('k array index')
    ax_delta.set_ylabel(
        r'$k_{allowed}/\sqrt{|K|}-k_{integer}/\sqrt{|K|}$')
    ax_delta.grid(True, alpha=0.25)

    # Classify sign changes of delta_k:
    #   dashed  — monotonic continuous crossing through 0
    #   dotted  — huge jump from negative to positive
    # Filters:
    #   Low-k wiggles  → short backward run-length → excluded.
    #   High-k near-0  → nothing large remains after the crossing → excluded.
    #     Forward max rather than local amplitude lets a crossing like index ~16
    #     (small local value, but large jumps still ahead) pass through.
    sign_changes = np.where(np.diff(np.sign(delta_k)))[0]
    abs_max        = np.max(np.abs(delta_k))
    jump_threshold = 0.3 * abs_max
    amp_min        = 0.1 * abs_max
    min_run        = 4               # minimum same-sign run before a smooth crossing

    smooth_crossings = []
    jump_crossings   = []
    for sc in sign_changes:
        y0, y1 = delta_k[sc], delta_k[sc + 1]
        x_cross = float(sc) - y0 / (y1 - y0)
        step = abs(y1 - y0)
        if step >= jump_threshold and y0 < 0 and y1 > 0:
            jump_crossings.append(x_cross)
        elif step < jump_threshold:
            # Filter 1 — high-k approaching-0: skip if nothing large remains
            # after this crossing (delta_k has settled to ~0 for good).
            future_max = np.max(np.abs(delta_k[sc + 1:])) if sc + 1 < len(delta_k) else 0.0
            if future_max < amp_min:
                continue
            # Filter 2 — low-k wiggles: require a minimum backward run-length.
            prev = sign_changes[sign_changes < sc]
            run_before = sc - prev[-1] if len(prev) > 0 else sc
            if run_before >= min_run:
                smooth_crossings.append(x_cross)

    for xc in smooth_crossings:
        ax_heat.axvline(xc, color='gray', lw=0.9, ls='--')
        ax_delta.axvline(xc, color='gray', lw=0.9, ls='--')
    for xc in jump_crossings:
        ax_heat.axvline(xc, color='gray', lw=0.9, ls=':')
        ax_delta.axvline(xc, color='gray', lw=0.9, ls=':')

    out = output_dir + 'cca_integerK_heatmap_with_delta_k.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


def plot_gram_comparison(G1_t, G2_t, G12_t, G1_2D, G2_2D, G12_2D,
                         output_dir=OUTPUT_DIR):
    """Compare eigenvalue spectra before and after applying spatial overlap."""
    _ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        (G1_t,  G1_2D,  'G1  (integerK self)'),
        (G2_t,  G2_2D,  'G2  (allowedK self)'),
        (G12_t, G12_2D, 'G12 (cross)'),
    ]

    for ax, (G_t, G_2D, title) in zip(axes, pairs):
        ev_t  = np.sort(np.abs(np.linalg.eigvalsh(G_t)))[::-1]                                                                                
        ev_2D = np.sort(np.abs(np.linalg.eigvalsh(G_2D)))[::-1]
        ev_t  = ev_t  / ev_t[0]
        ev_2D = ev_2D / ev_2D[0]
        ax.semilogy(ev_t,  'b-o', ms=3, label='time-only')
        ax.semilogy(ev_2D, 'r-s', ms=3, label='2D space-time')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Index')
        ax.set_ylabel('λ_i / λ_max')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Gram eigenvalue spectra: time-only vs 2D space-time', fontsize=11)
    fig.tight_layout()
    out = output_dir + '2D_gram_comparison.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()



# ============================================================================
# Main analysis function
# ============================================================================

def cca_sparse_2D_reg_analysis(
        N_t               = 1000,
        alpha             = 1e-6,      # k-proportional penalty strength
        weights           = None,
        normalise_spatial = True,
        ev_threshold      = 1e-10,
        rho_min           = 0.99,
        rotation_method   = 'promax',
        folder_path       = FOLDER_PATH,
        output_dir        = OUTPUT_DIR,
        results_cache     = RESULTS_CACHE,
        force_recompute   = False,
):
    """
    2D space-time CCA with k-proportional diagonal penalty (continuous).

    Parameters
    ----------
    N_t               : number of time-grid points
    alpha             : k-proportional penalty strength (G_reg = G + alpha*diag(k))
    weights           : per-perturbation-type weights {p: float}; None = unit
    normalise_spatial : if True, normalise S matrices so diagonal = 1
    ev_threshold      : relative eigenvalue cutoff for G^{-1/2}
    rho_min           : minimum canonical correlation to retain a mode
    rotation_method   : 'varimax' | 'promax' | 'ica'
    folder_path       : path to data/ directory
    output_dir        : directory for output figures
    results_cache     : pickle file for caching results
    force_recompute   : ignore cache if True

    Returns
    -------
    dict with all results:
        A, B, A_sparse, B_sparse, rho, rho_all, alpha, ...
    """

    if not force_recompute and os.path.exists(results_cache):
        print(f"Loading cached results from {results_cache}")
        with open(results_cache, 'rb') as f:
            return pickle.load(f)

    # ── Step 1: Load bases ────────────────────────────────────────────────────
    basis_1, basis_2, eta_grid = load_bases(folder_path, N_t)

    # ── Step 1b: Load k-values ────────────────────────────────────────────────
    print("\n--- Step 1b: Loading k-values ---")
    k_values_1 = load_kvalues('integerK', folder_path)
    k_values_2 = load_kvalues('allowedK', folder_path)
    print(f"  integerK: {len(k_values_1)} modes")
    print(f"  allowedK: {len(k_values_2)} modes")

    beta1 = k_to_beta(k_values_1)
    beta2 = k_to_beta(k_values_2)
    print_beta_info(beta1, beta2)

    # ── Step 2: Frobenius normalise ───────────────────────────────────────────
    print("\n--- Step 2: Frobenius normalisation ---")
    Xn1, norms_1 = frobenius_normalize(basis_1)
    Xn2, norms_2 = frobenius_normalize(basis_2)

    # ── Step 3a: Time-only Gram matrices ──────────────────────────────────────
    print("\n--- Step 3a: Time-only Gram matrices ---")
    G1_t, G2_t, G12_t = compute_time_gram_matrices(Xn1, Xn2, weights)
    print(f"  G1_t shape: {G1_t.shape},  G12_t shape: {G12_t.shape}")

    # ── Step 3b: Spatial overlap matrices ────────────────────────────────────
    print("\n--- Step 3b: Spatial overlap matrices ---")
    print("  Computing S11 (integerK self-overlap)...")
    S11 = compute_spatial_overlap(beta1, beta1)
    print("  Computing S22 (allowedK self-overlap)...")
    S22 = compute_spatial_overlap(beta2, beta2)
    print("  Computing S12 (cross-overlap)...")
    S12 = compute_spatial_overlap(beta1, beta2)
    n12 = min(len(beta1), len(beta2))
    print(f"  S11 diagonal range: [{np.diag(S11).min():.3e}, {np.diag(S11).max():.3e}]")
    print(f"  S11 off-diag max:   {(np.abs(S11) - np.diag(np.abs(np.diag(S11)))).max():.3e}")
    print(f"  S12 diag approx range: "
          f"[{np.diag(S12[:n12, :n12]).min():.3e}, "
          f"{np.diag(S12[:n12, :n12]).max():.3e}]")

    # ── Step 3c: 2D Gram matrices ─────────────────────────────────────────────
    print(f"\n--- Step 3c: 2D space-time Gram matrices "
          f"(normalise_spatial={normalise_spatial}) ---")
    G1_2D, G2_2D, G12_2D = compute_2d_gram_matrices(
        G1_t, G2_t, G12_t, S11, S22, S12,
        normalise_spatial=normalise_spatial)
    print(f"  G1_2D shape: {G1_2D.shape}")

    # ── Step 4: k-proportional rCCA ───────────────────────────────────────────
    print(f"\n--- Step 4: k-proportional rCCA  (alpha = {alpha}) ---")

    A, B, rho, rho_all = solve_rcca_k_scaled(
        G1_2D, G2_2D, G12_2D, k_values_1, k_values_2, alpha, ev_threshold, rho_min)

    print(f"  Valid modes found: {len(rho)}")

    # ── Step 5: Sparse rotation ───────────────────────────────────────────────
    if A.shape[1] > 0:
        print("\n--- Step 5: Sparse rotation ---")
        # Fit one balanced rotation to both bases, rather than choosing the
        # sparse orientation from integerK alone.
        A_sparse, B_sparse = apply_joint_sparse_rotation(
            A, B, method=rotation_method)

        g1 = np.mean([gini(A_sparse[:, i]) for i in range(A_sparse.shape[1])])
        g2 = np.mean([gini(B_sparse[:, i]) for i in range(B_sparse.shape[1])])
        print(f"\n  Mean Gini — Basis 1: {g1:.3f}, Basis 2: {g2:.3f}")
    else:
        print("  No valid modes found — skipping rotation.")
        A_sparse = A.copy()
        B_sparse = B.copy()

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_spatial_overlap(S12, beta1, beta2, output_dir)
    plot_gram_comparison(G1_t, G2_t, G12_t, G1_2D, G2_2D, G12_2D, output_dir)
    plot_gram_spectrum(G1_2D, G2_2D, output_dir)

    if len(rho_all) > 0:
        plot_canonical_correlations(rho_all, rho_min, output_dir)

    if A.shape[1] > 0:
        plot_coefficients(A, B, A_sparse, B_sparse, rho,
                          N_plot=min(20, A.shape[1]), output_dir=output_dir)

    plot_heatmap(A_sparse, B_sparse, output_dir)
    plot_integer_heatmap_with_k_difference(
        A_sparse, k_values_1, k_values_2, output_dir)

    if A_sparse.shape[1] > 0:
        plot_reconstructed_timeseries(
            basis_1, basis_2,
            A_sparse, B_sparse, rho,
            eta_grid, norms_1, norms_2,
            N_plot=4, output_dir=output_dir,
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    results = dict(
        eta_grid          = eta_grid,
        G1_t=G1_t, G2_t=G2_t, G12_t=G12_t,
        G1_2D=G1_2D, G2_2D=G2_2D, G12_2D=G12_2D,
        S11=S11, S22=S22, S12=S12,
        A                 = A,
        B                 = B,
        A_sparse          = A_sparse,
        B_sparse          = B_sparse,
        rho               = rho,
        rho_all           = rho_all,
        basis_1           = basis_1,
        basis_2           = basis_2,
        norms_1           = norms_1,
        norms_2           = norms_2,
        k_values_1        = k_values_1,
        k_values_2        = k_values_2,
        beta1             = beta1,
        beta2             = beta2,
        alpha             = alpha,
        weights           = weights,
        rotation_method   = rotation_method,
        normalise_spatial = normalise_spatial,
        ev_threshold      = ev_threshold,
        rho_min           = rho_min,
    )
    with open(results_cache, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_cache}")
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    weights = {
        'dr': 1.0,
        'dm': 100.0,
        'vr': 1.0,
        'vm': 1.0,
    }

    results = cca_sparse_2D_reg_analysis(
        N_t               = 1000,
        alpha             = 1e-8,
        weights           = weights,
        normalise_spatial = True,
        ev_threshold      = 1e-10,
        rho_min           = 0.99,
        rotation_method   = 'promax',
        folder_path       = FOLDER_PATH,
        force_recompute   = True,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  alpha             : {results['alpha']}")
    print(f"  Total valid modes : {results['A_sparse'].shape[1]}")
    print(f"  N_k (integerK)    : {results['G1_2D'].shape[0]}")
    print(f"  N_k (allowedK)    : {results['G2_2D'].shape[0]}")
    print("=" * 60)
