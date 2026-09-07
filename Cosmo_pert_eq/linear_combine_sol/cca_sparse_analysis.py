# -*- coding: utf-8 -*-
"""
CCA-Based Sparse Joint Eigenfunction Method

Pipeline (see cca_formulation/cca_formulation.pdf for full derivation):
  1. Load integerK (basis 1) and allowedK (basis 2) perturbation solutions.
  2. Frobenius-normalise each perturbation type.
  3. Compute k-space Gram matrices G1, G2, G12  (each N_k x N_k).
  4. Whitened SVD:  G1^{-1/2} @ G12 @ G2^{-1/2} = U S V^T
     Singular values rho_i in [0,1] are the canonical correlations.
  5. Canonical vectors: a_i = G1^{-1/2} @ u_i,  b_i = G2^{-1/2} @ v_i
     Keep m modes with rho_i > rho_min.
  6. Apply Promax oblique rotation (or Varimax / FastICA) to A = [a_1,...,a_m]
     to find maximally sparse linear combinations; apply same transform to B.

Key differences from svd_sparse_analysis.py:
  - Works entirely in k-space: Gram matrices are (N_k, N_k) vs (4*N_t, N_k).
  - rho_i in [0,1] with rho_i=1 <=> exact joint eigenfunction (no mixing
    matrix step needed).
  - G^{-1/2} whitening corrects per-k-mode amplitude bias that Frobenius
    normalisation alone does not remove (see cca_formulation.pdf §3.3).
  - Two separate coefficient vectors per mode: a_i (basis 1), b_i (basis 2).

See cca_formulation/cca_formulation.pdf for full mathematical derivation.
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../multi_pert_sparse_ortho')

from multi_perturbation_analysis import generate_multi_perturbation_bases, fcb_time

# ---------------------------------------------------------------------------
PERTURBATION_TYPES = ['dr', 'dm', 'vr', 'vm']
FOLDER_PATH        = './data/'
OUTPUT_DIR         = './figures/'
RESULTS_CACHE      = 'cca_sparse_results.pickle'
# ---------------------------------------------------------------------------


# ============================================================================
# STEP 1 — Load perturbation bases
# ============================================================================

def load_bases(folder_path=FOLDER_PATH, N_t=1000):
    """Load integerK and allowedK bases onto a uniform eta grid."""
    eta_grid = np.linspace(0, fcb_time, N_t)
    print("Loading integerK (basis 1)...")
    basis_1 = generate_multi_perturbation_bases("integerK", eta_grid,
                                                folder_path=folder_path)
    print("\nLoading allowedK (basis 2)...")
    basis_2 = generate_multi_perturbation_bases("allowedK", eta_grid,
                                                folder_path=folder_path)
    return basis_1, basis_2, eta_grid


# ============================================================================
# STEP 2 — Frobenius normalise
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
# STEP 3 — Gram matrices
# ============================================================================

def compute_gram_matrices(Xn1, Xn2, weights=None):
    """
    Compute the three k-space Gram matrices (eq. 5-7 in cca_formulation.pdf):

        G1  = sum_p  w_p  (Xn1^p)^T  Xn1^p    shape (N_k, N_k)
        G2  = sum_p  w_p  (Xn2^p)^T  Xn2^p
        G12 = sum_p  w_p  (Xn1^p)^T  Xn2^p

    Parameters
    ----------
    Xn1, Xn2 : dicts of Frobenius-normalised (N_t, N_k) arrays
    weights  : dict {p: float} or None (unit weights w_p = 1)

    Returns
    -------
    G1, G2, G12  each of shape (N_k, N_k)
    """
    if weights is None:
        weights = {p: 1.0 for p in PERTURBATION_TYPES}

    N_k = Xn1[PERTURBATION_TYPES[0]].shape[1]
    G1  = np.zeros((N_k, N_k))
    G2  = np.zeros((N_k, N_k))
    G12 = np.zeros((N_k, N_k))

    for p in PERTURBATION_TYPES:
        w   = weights[p]
        X1p = Xn1[p]   # (N_t, N_k)
        X2p = Xn2[p]
        G1  += w * X1p.T @ X1p
        G2  += w * X2p.T @ X2p
        G12 += w * X1p.T @ X2p

    return G1, G2, G12


# ============================================================================
# STEP 4a — Matrix square-root inverse  G^{-1/2}
# ============================================================================

def matrix_sqrt_inv(G, ev_threshold=1e-10):
    """
    Compute G^{-1/2} via eigendecomposition with rank truncation.

        G = Q Lambda Q^T  =>  G^{-1/2} = Q Lambda^{-1/2} Q^T

    Small eigenvalues (< ev_threshold * lambda_max) are discarded to avoid
    numerical instability from near-singular Gram matrices.

    Parameters
    ----------
    ev_threshold : relative cutoff; eigenvalues < threshold * lambda_max dropped

    Returns
    -------
    Ginvhalf : (N_k, N_k) symmetric pseudo-inverse square root
    rank     : number of eigenvalues kept
    """
    G = 0.5 * (G + G.T)                          # symmetrise
    eigvals, eigvecs = np.linalg.eigh(G)          # ascending order

    cutoff = ev_threshold * eigvals[-1]
    keep   = eigvals > cutoff
    rank   = int(np.sum(keep))

    lam      = eigvals[keep]
    Q        = eigvecs[:, keep]                   # (N_k, rank)
    Ginvhalf = Q @ np.diag(1.0 / np.sqrt(lam)) @ Q.T
    return Ginvhalf, rank


# ============================================================================
# STEP 4b — Whitened SVD => canonical correlations and vectors
# ============================================================================

def solve_cca(G1, G2, G12, ev_threshold=1e-10, rho_min=0.99):
    """
    Solve CCA via SVD of the whitened cross-Gram matrix (eq. 8-12 in pdf):

        W = G1^{-1/2} @ G12 @ G2^{-1/2} = U S V^T

    Canonical correlations rho_i = S_i in [0, 1].
    Canonical vectors:
        a_i = G1^{-1/2} @ u_i    (basis-1 k-space coefficients)
        b_i = G2^{-1/2} @ v_i    (basis-2 k-space coefficients)

    Parameters
    ----------
    ev_threshold : relative threshold for G^{-1/2} rank truncation
    rho_min      : keep modes with canonical correlation rho_i > rho_min

    Returns
    -------
    A       (N_k, m)  basis-1 canonical vectors
    B       (N_k, m)  basis-2 canonical vectors
    rho     (m,)      canonical correlations of kept modes
    rho_all (r,)      all canonical correlations (for plotting)
    """
    print("  Computing G1^{-1/2} ...")
    G1inv, rank1 = matrix_sqrt_inv(G1, ev_threshold)
    print(f"    rank(G1) = {rank1} / {G1.shape[0]}")

    print("  Computing G2^{-1/2} ...")
    G2inv, rank2 = matrix_sqrt_inv(G2, ev_threshold)
    print(f"    rank(G2) = {rank2} / {G2.shape[0]}")

    # Whitened cross-Gram
    W = G1inv @ G12 @ G2inv                      # (N_k, N_k)

    # SVD: W = U S V^T
    U, rho_all, Vt = np.linalg.svd(W, full_matrices=False)
    rho_all = np.clip(rho_all, 0.0, 1.0)         # clip numerical noise

    valid = rho_all > rho_min
    m     = int(np.sum(valid))
    print(f"  Valid modes (rho > {rho_min}): {m} / {len(rho_all)}")
    if m > 0:
        print(f"  rho range: {rho_all[valid].min():.6f} – {rho_all[valid].max():.6f}")

    U_valid = U[:, valid]                         # (N_k, m)
    V_valid = Vt[valid, :].T                      # (N_k, m)

    A = G1inv @ U_valid                           # basis-1 k-space coefficients
    B = G2inv @ V_valid                           # basis-2 k-space coefficients

    return A, B, rho_all[valid], rho_all


# ============================================================================
# STEP 5 — Sparse rotation  (identical logic to svd_sparse_analysis.py)
# ============================================================================

def varimax_rotation(Phi, gamma=1.0, q=500, tol=1e-6):
    """
    Varimax orthogonal rotation on Phi (N x K). Returns (Phi_rotated, R).
    """
    p, k = Phi.shape
    rs   = np.random.RandomState(42)
    R    = np.linalg.qr(rs.randn(k, k))[0]
    d    = 0.0
    for i in range(q):
        d_old = d
        B     = Phi @ R
        M2    = B ** 2
        u, s, vh = np.linalg.svd(
            Phi.T @ (B * M2
                     - (gamma / p) * B @ np.diag(M2.sum(axis=0)))
        )
        R = u @ vh
        d = s.sum()
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            print(f"  Varimax converged in {i+1} iterations")
            return Phi @ R, R
    print(f"  Varimax reached max iterations ({q})")
    return Phi @ R, R


def promax_rotation(Phi, power=3, gamma=1.0, q=500, tol=1e-6):
    """
    Promax oblique rotation on Phi (N x K). Returns (pattern, R_v, L_norm).
    """
    F_v, R_v  = varimax_rotation(Phi, gamma=gamma, q=q, tol=tol)
    T         = np.sign(F_v) * (np.abs(F_v) ** power)
    L, _, _, _ = np.linalg.lstsq(F_v, T, rcond=None)
    col_norms = np.linalg.norm(L, axis=0)
    col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
    L_norm    = L / col_norms
    return F_v @ L_norm, R_v, L_norm


def ica_rotation(A, B):
    """FastICA on A; apply same rotation to B. Returns A_sparse, B_sparse."""
    from sklearn.decomposition import FastICA
    m    = A.shape[1]
    ica  = FastICA(n_components=m, random_state=42, max_iter=2000, tol=1e-5)
    A_sp = ica.fit_transform(A)
    W, _, _, _ = np.linalg.lstsq(A, A_sp, rcond=None)
    B_sp = B @ W
    print("  FastICA done; rotation recovered from A.")
    return A_sp, B_sp


def apply_sparse_rotation(A, B, method='promax'):
    """
    Sparsify A via oblique rotation; apply same transform to B.
    method : 'varimax' | 'promax' | 'ica'
    Returns A_sparse, B_sparse  each (N_k, m).
    """
    print(f"\nApplying {method} rotation...")
    if method == 'ica':
        return ica_rotation(A, B)
    if method == 'promax':
        A_sparse, R_v, L_norm = promax_rotation(A)
        B_sparse = (B @ R_v) @ L_norm
        return A_sparse, B_sparse
    # Varimax
    A_sparse, R = varimax_rotation(A)
    B_sparse    = B @ R
    return A_sparse, B_sparse


def apply_joint_sparse_rotation(A, B, method='promax'):
    """Sparsify both CCA bases using one jointly fitted rotation.

    The rotation is fitted to a vertically stacked, Frobenius-balanced copy
    of ``A`` and ``B`` and then applied to the original coefficients. This
    treats the two matched bases symmetrically while preserving mode pairing.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
        raise ValueError(
            f"A and B must be two-dimensional arrays with equal shape; "
            f"got {A.shape} and {B.shape}"
        )
    if A.shape[1] == 0:
        return A.copy(), B.copy()

    norm_A = np.linalg.norm(A, 'fro')
    norm_B = np.linalg.norm(B, 'fro')
    if norm_A < 1e-30 or norm_B < 1e-30:
        raise ValueError("Cannot fit a joint rotation to a zero-norm basis")

    joint = np.vstack((A / norm_A, B / norm_B))
    print(f"\nApplying joint {method} rotation to both bases...")
    if method == 'promax':
        _, R_v, L_norm = promax_rotation(joint)
        transform = R_v @ L_norm
    elif method == 'varimax':
        _, transform = varimax_rotation(joint)
    else:
        raise ValueError(
            "Joint sparse rotation supports 'varimax' and 'promax'; "
            f"got {method!r}"
        )
    return A @ transform, B @ transform


# ============================================================================
# Utilities
# ============================================================================

def gini(v):
    """Gini coefficient of |v|. Range [0,1]; 1 = perfectly sparse."""
    v = np.sort(np.abs(v.ravel()))
    n = len(v)
    s = v.sum()
    if s < 1e-30:
        return 0.0
    return 1.0 - 2.0 * np.dot(v, n - np.arange(n)) / (n * s)


def normalize_columns(C):
    norms = np.linalg.norm(C, axis=0)
    norms = np.where(norms < 1e-30, 1.0, norms)
    return C / norms


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ============================================================================
# Plotting
# ============================================================================

def plot_gram_spectrum(G1, G2, output_dir=OUTPUT_DIR):
    """Eigenvalue spectra of G1 and G2 — diagnostic for rank truncation."""
    _ensure_dir(output_dir)
    ev1 = np.sort(np.linalg.eigvalsh(G1))[::-1]
    ev2 = np.sort(np.linalg.eigvalsh(G2))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ev, label in zip(axes,
                              [ev1, ev2],
                              ['G1 (integerK)', 'G2 (allowedK)']):
        ax.semilogy(np.arange(1, len(ev)+1), ev / ev[0], 'o-', ms=3)
        ax.set_xlabel('Index')
        ax.set_ylabel('λ_i / λ_max')
        ax.set_title(f'Gram eigenvalue spectrum: {label}')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir + 'cca_gram_spectrum.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_canonical_correlations(rho_all, rho_min, output_dir=OUTPUT_DIR):
    """Bar plot of all canonical correlations with threshold line."""
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(rho_all)+1), rho_all, 'o-', ms=4)
    ax.axhline(rho_min, color='r', ls='--', label=f'threshold = {rho_min}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Canonical correlation ρ')
    ax.set_ylim(0, 1.05)
    ax.set_title('CCA: canonical correlations between Basis 1 and Basis 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir + 'cca_canonical_correlations.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_coefficients(A, B, A_sparse, B_sparse, rho,
                      N_plot=10, output_dir=OUTPUT_DIR):
    """Four-column bar plot: orig B1 | orig B2 | sparse B1 | sparse B2."""
    _ensure_dir(output_dir)
    N_k    = A.shape[0]
    N_plot = min(N_plot, A.shape[1])

    dom1  = np.argmax(np.abs(A_sparse), axis=0)
    order = np.argsort(dom1)

    fig, axes = plt.subplots(N_plot, 4, figsize=(14, 1.8 * N_plot))
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("k-space coefficients — original vs sparse (both bases)", fontsize=11)

    for row, idx in enumerate(order[:N_plot]):
        d_list = [A[:, idx].real, B[:, idx].real,
                  A_sparse[:, idx].real, B_sparse[:, idx].real]
        t_list = [f"B1 orig  ρ={rho[idx]:.4f}", "B2 orig",
                  f"B1 sparse  dom k={dom1[idx]+1}", "B2 sparse"]
        c_list = ['steelblue', 'seagreen', 'steelblue', 'seagreen']

        for col, (d, t, c) in enumerate(zip(d_list, t_list, c_list)):
            ax = axes[row, col]
            ax.bar(np.arange(1, N_k + 1), d, color=c, alpha=0.7, width=0.6)
            ax.set_title(t, fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(labelsize=7)
            ax.label_outer()

    for col, lbl in enumerate(["k index (B1)", "k index (B2)"] * 2):
        axes[-1, col].set_xlabel(lbl, fontsize=8)

    fig.tight_layout()
    out = output_dir + 'cca_coefficients.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_heatmap(A_sparse, B_sparse, output_dir=OUTPUT_DIR):
    """Heatmap of column-normalised |coefficients| sorted by dominant k in basis 1."""
    _ensure_dir(output_dir)
    A_norm = normalize_columns(A_sparse)
    B_norm = normalize_columns(B_sparse)
    dom1   = np.argmax(np.abs(A_norm), axis=0)
    order  = np.argsort(dom1)
    A_s    = A_norm[:, order]
    B_s    = B_norm[:, order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(np.abs(A_s.T), aspect='auto', cmap='Blues',
                     interpolation='nearest', vmin=0, vmax=1)
    ax1.set_xlabel('k index (integerK)')
    ax1.set_ylabel('Mode (sorted by dominant k)')
    ax1.set_title('|Normalised sparse coefficients| — Basis 1 (CCA)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(np.abs(B_s.T), aspect='auto', cmap='Greens',
                     interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('k index (allowedK)')
    ax2.set_ylabel('Mode (sorted by dominant k)')
    ax2.set_title('|Normalised sparse coefficients| — Basis 2 (CCA)')
    plt.colorbar(im2, ax=ax2)

    fig.suptitle('Joint Sparsity Heatmap (CCA method, column-normalised)', fontsize=12)
    fig.tight_layout()
    out = output_dir + 'cca_heatmap.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


def plot_reconstructed_timeseries(basis_1, basis_2,
                                  A_sparse, B_sparse, rho,
                                  eta_grid, norms_1, norms_2,
                                  N_plot=5, output_dir=OUTPUT_DIR):
    """
    Overlay time series from both bases using sparse coefficients.

    Divides by Frobenius norms so the reconstruction corresponds to
    Xn^p @ coeff (the normalised space in which the CCA was solved),
    avoiding spurious per-field scale factors.

    The final eta sample is omitted only for ``vm``, whose numerical velocity
    reconstruction can contain a non-physical spike exactly at the FCB.
    """
    _ensure_dir(output_dir)
    dom1   = np.argmax(np.abs(A_sparse), axis=0)
    order  = np.argsort(dom1)
    N_plot = min(N_plot, A_sparse.shape[1])

    if len(eta_grid) < 2:
        raise ValueError("eta_grid must contain at least two samples")
    perturbation_labels = {
        'dr': r'$\delta_r$',
        'dm': r'$\delta_m$',
        'vr': r'$v_r$',
        'vm': r'$v_m$',
    }

    # Keep the PDF narrower than the original 14-inch layout so it scales
    # reliably inside both one- and two-column LaTeX figures.
    fig, axes = plt.subplots(N_plot, 4, figsize=(9, 1.5 * N_plot),
                             constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("CCA sparse modes: time-series (Basis 1 red vs Basis 2 green, "
                 "Frobenius-normalised)", fontsize=11)

    for row, idx in enumerate(order[:N_plot]):
        dom_k = dom1[idx]
        for col, p in enumerate(PERTURBATION_TYPES):
            ax = axes[row, col]
            plot_slice = slice(None, -1) if p == 'vm' else slice(None)
            eta_plot = eta_grid[plot_slice]
            s1 = (basis_1[p] @ A_sparse[:, idx] / norms_1[p])[plot_slice]
            s2 = (basis_2[p] @ B_sparse[:, idx] / norms_2[p])[plot_slice]
            if np.dot(s1, s2) < 0:
                s2 = -s2
            ax.plot(eta_plot, s1, 'r-',  lw=2,   alpha=0.85, label='B1')
            ax.plot(eta_plot, s2, 'g--', lw=1.5, alpha=0.85, label='B2')
            ax.set_title(
                f"{perturbation_labels.get(p, p)}  "
                rf"$k_\text{{dom}}$ idx={dom_k + 1}  ρ={rho[idx]:.4f}",
                fontsize=8,
            )
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            if row == N_plot - 1:
                ax.set_xlabel("η", fontsize=9)

    out = output_dir + 'cca_timeseries.pdf'
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Main analysis function
# ============================================================================

def cca_sparse_analysis(
        N_t             = 1000,
        weights         = None,       # dict {p: float} or None for unit weights
        ev_threshold    = 1e-10,      # relative threshold for G^{-1/2} rank truncation
        rho_min         = 0.99,       # canonical correlation threshold
        rotation_method = 'promax',   # 'varimax' | 'promax' | 'ica'
        folder_path     = FOLDER_PATH,
        output_dir      = OUTPUT_DIR,
        results_cache   = RESULTS_CACHE,
        force_recompute = False,
):
    """
    Full CCA-based sparse joint eigenfunction analysis.

    Parameters
    ----------
    N_t             : number of time-grid points
    weights         : per-perturbation-type weights {p: float}; None = unit weights
    ev_threshold    : relative eigenvalue cutoff for G^{-1/2} computation
    rho_min         : minimum canonical correlation to count as a valid mode
    rotation_method : sparsifying rotation — 'varimax', 'promax', or 'ica'
    folder_path     : path to data/ directory
    output_dir      : directory for output figures
    results_cache   : pickle file to cache/reload results
    force_recompute : ignore cache if True

    Returns
    -------
    dict with keys: eta_grid, G1, G2, G12, A, B, A_sparse, B_sparse,
                    rho, rho_all, basis_1, basis_2, norms_1, norms_2, ...
    """

    if not force_recompute and os.path.exists(results_cache):
        print(f"Loading cached results from {results_cache}")
        with open(results_cache, 'rb') as f:
            return pickle.load(f)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    basis_1, basis_2, eta_grid = load_bases(folder_path, N_t)

    # ── Step 2: Frobenius normalise ───────────────────────────────────────────
    print("\n--- Step 2: Frobenius normalisation ---")
    Xn1, norms_1 = frobenius_normalize(basis_1)
    Xn2, norms_2 = frobenius_normalize(basis_2)

    # ── Step 3: Gram matrices ─────────────────────────────────────────────────
    print("\n--- Step 3: Gram matrices ---")
    G1, G2, G12 = compute_gram_matrices(Xn1, Xn2, weights)
    print(f"  Gram matrix shape: {G1.shape}")

    # ── Step 4: CCA ───────────────────────────────────────────────────────────
    print("\n--- Step 4: CCA (whitened SVD) ---")
    A, B, rho, rho_all = solve_cca(G1, G2, G12, ev_threshold, rho_min)
    print(f"  A shape: {A.shape},  B shape: {B.shape}")

    # ── Step 5: Sparse rotation ───────────────────────────────────────────────
    print("\n--- Step 5: Sparse rotation ---")
    A_sparse, B_sparse = apply_sparse_rotation(A, B, method=rotation_method)

    # ── Sparsity metrics ──────────────────────────────────────────────────────
    g_before_1 = np.mean([gini(A[:, i]) for i in range(A.shape[1])])
    g_after_1  = np.mean([gini(A_sparse[:, i]) for i in range(A_sparse.shape[1])])
    g_before_2 = np.mean([gini(B[:, i]) for i in range(B.shape[1])])
    g_after_2  = np.mean([gini(B_sparse[:, i]) for i in range(B_sparse.shape[1])])
    print(f"\nMean Gini sparsity (higher = sparser):")
    print(f"  Basis 1 — before: {g_before_1:.3f},  after: {g_after_1:.3f}")
    print(f"  Basis 2 — before: {g_before_2:.3f},  after: {g_after_2:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n--- Plotting ---")
    plot_gram_spectrum(G1, G2, output_dir)
    plot_canonical_correlations(rho_all, rho_min, output_dir)
    n_show = min(20, len(rho))
    plot_coefficients(A, B, A_sparse, B_sparse, rho,
                      N_plot=n_show, output_dir=output_dir)
    plot_heatmap(A_sparse, B_sparse, output_dir)
    plot_reconstructed_timeseries(basis_1, basis_2,
                                  A_sparse, B_sparse, rho,
                                  eta_grid, norms_1, norms_2,
                                  N_plot=5, output_dir=output_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    results = dict(
        eta_grid        = eta_grid,
        G1=G1, G2=G2, G12=G12,
        A=A, B=B,
        A_sparse        = A_sparse,
        B_sparse        = B_sparse,
        rho             = rho,
        rho_all         = rho_all,
        basis_1         = basis_1,
        basis_2         = basis_2,
        norms_1         = norms_1,
        norms_2         = norms_2,
        weights         = weights,
        rotation_method = rotation_method,
        ev_threshold    = ev_threshold,
        rho_min         = rho_min,
    )
    with open(results_cache, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_cache}")
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # Per-field weights to equalise contributions to the CCA objective.
    # After Frobenius normalisation the Gram matrices have equal total energy,
    # but the CCA can still find directions where low-amplitude fields (dm, dr)
    # have negligible projected signal and are effectively ignored.
    # Heuristic: w_p = (max_amp / amp_p)^2 where amp_p is the typical
    # projected amplitude per field (vr is the largest at ~0.075).
    weights = {
        'dr': 1.0,      # (vr_amp/dr_amp)^2 ≈ (0.075/0.04)^2 ≈ 4
        'dm': 100.0,   # (vr_amp/dm_amp)^2 ≈ (0.075/0.001)^2 ≈ 5600; conservative start
        'vr': 1.0,      # baseline (largest projected amplitude)
        'vm': 1.0,
    }

    results = cca_sparse_analysis(
        N_t             = 1000,
        weights         = weights,
        ev_threshold    = 1e-10,
        rho_min         = 0.99,
        rotation_method = 'promax',   # 'varimax' | 'promax' | 'ica'
        force_recompute = True,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Valid modes found : {len(results['rho'])}")
    print(f"  rho_valid         : {np.round(results['rho'], 6)}")
    print(f"  N_k               : {results['G1'].shape[0]}")
    print("=" * 60)
