import gc
import os
import numpy as np
import time
from multiprocessing import Pool

start_time = time.time()

data_dir = "data/diff_m/"
os.makedirs(data_dir, exist_ok=True)

T = np.pi
L = 2.0 * np.pi
eigenvalues_threshold = 0.01
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
kappa_list = np.linspace(0.01, 1, num=100)


def gram_schmidt_rows(F):
    """Orthonormalize rows of F using modified Gram-Schmidt.
    Returns Q (same shape, orthonormal rows) and T (N×N, complex) where Q = T @ F."""
    N = F.shape[0]
    Q = np.empty_like(F)
    T = np.zeros((N, N), dtype=complex)
    for i in range(N):
        q = F[i].copy()
        T[i, i] = 1.0
        for j in range(i):
            coeff = np.vdot(Q[j], q)
            q -= coeff * Q[j]
            T[i] -= coeff * T[j]
        norm = np.sqrt(np.vdot(q, q).real)
        Q[i] = q / norm
        T[i] /= norm
    return Q, T


def choose_eigenvalues(eigenvalues, threshold):
    return [i for i in range(len(eigenvalues)) if np.abs(eigenvalues[i] - 1.0) < threshold]


def solve_for_kappa(kappa, N, Nt):
    # Grids with exact discrete orthogonality properties
    x_list = np.linspace(0, L, Nt, endpoint=False)  # no endpoint → exact exp(ikx) orth.
    dt     = T / Nt
    t_list = np.linspace(dt / 2, T - dt / 2, Nt)   # staggered → exact cos(nt) orth.
    t, x   = np.meshgrid(t_list, x_list)

    # Build basis-1 directly into a stacked matrix F1 of shape (N, Nt*Nt).
    #
    # WHY: the previous code built a Python list of N separate (Nt, Nt) arrays,
    # then called np.vdot N² times in a Python loop to fill the overlap matrix M.
    # That approach has two problems:
    #   1. Memory: the list doubles peak RAM (basis + orthonormal copy live together).
    #   2. Speed: N² Python-level vdot calls scale as O(N² × Nt²) with large overhead.
    #
    # By filling a pre-allocated (N, Nt²) matrix row-by-row we keep only one copy
    # in memory, then compute M = F1.conj() @ F2.T with a single BLAS call — roughly
    # N× faster than the loop and half the peak memory.
    flat      = Nt * Nt
    F1        = np.empty((N, flat), dtype=complex)
    inv_norm1 = np.empty(N)
    for i, k in enumerate(range(1, N + 1)):
        f            = (np.cos(np.sqrt(k**2 + kappa**2) * t) * np.exp(1j * k * x)).ravel()
        norm         = np.sqrt(np.vdot(f, f).real)
        inv_norm1[i] = 1.0 / norm
        F1[i]        = f / norm

    start_n  = int(kappa) + 1
    F2_raw   = np.empty((N, flat), dtype=complex)
    for i, n in enumerate(range(start_n, N + start_n)):
        kx         = np.sqrt((n * np.pi / T)**2 - kappa**2)
        F2_raw[i]  = (np.cos(n * np.pi / T * t) * np.exp(1j * kx * x)).ravel()

    # basis_2 has irrational x-wavenumbers so it is NOT orthogonal on the grid;
    # GS is required to get the true subspace-overlap matrix.
    F2, T2 = gram_schmidt_rows(F2_raw)
    del F2_raw

    del t, x, t_list, x_list   # free meshgrid — no longer needed

    # Single BLAS matrix multiply: M[i,j] = <F1[i], F2[j]>
    M = F1.conj() @ F2.T        # (N, N)
    del F1, F2

    # SVD: singular values S satisfy eigenvalues of MM† = S²
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    del M
    eigenvalues    = S ** 2
    eigenvectors_1 = U             # columns = left  singular vectors
    eigenvectors_2 = Vh.T.conj()   # columns = right singular vectors

    valid_idx = choose_eigenvalues(eigenvalues, eigenvalues_threshold)
    ev_valid  = eigenvalues[valid_idx]

    # Coefficients in the original (un-normalised) basis.
    # The transformation matrix is diagonal (1/norm per mode), so the full
    # matrix multiply reduces to element-wise row scaling.
    ev1_T          = eigenvectors_1[:, valid_idx].T   # (n_valid, N)
    ev2_T          = eigenvectors_2[:, valid_idx].T
    coefficients_1 = ev1_T * inv_norm1[np.newaxis, :]   # T1 = diag(inv_norm1) since F1 is already orthogonal
    coefficients_2 = ev2_T @ T2                          # T2 from GS: Q = T2 @ F2_raw
    del T2

    tag = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.3f}"
    np.savetxt(os.path.join(data_dir, f"all_eigenvalues1_2d_{tag}.txt"), eigenvalues)
    np.savetxt(os.path.join(data_dir, f"all_eigenvalues2_2d_{tag}.txt"), eigenvalues)
    np.savetxt(os.path.join(data_dir, f"eigenvalues1_2d_{tag}.txt"),     ev_valid)
    np.savetxt(os.path.join(data_dir, f"eigenvalues2_2d_{tag}.txt"),     ev_valid)
    np.savetxt(os.path.join(data_dir, f"coefficients1_2d_{tag}.txt"),    coefficients_1)
    np.savetxt(os.path.join(data_dir, f"coefficients2_2d_{tag}.txt"),    coefficients_2)

    print(f"  saved: {tag}  valid={len(ev_valid)}", flush=True)


def _worker(kappa):

    N  = max(30, int(kappa * 150))
    Nt = 5 * N

    tag      = f"N{N}_Nt{Nt}_T{T:.2f}_m{kappa:.3f}"
    out_file = os.path.join(data_dir, f"coefficients1_2d_{tag}.txt")
    if os.path.exists(out_file):
        print(f"kappa={kappa:.3f}: already computed, skipping", flush=True)
        return

    print(f"kappa={kappa:.3f}: N={N}, Nt={Nt} ...", flush=True)
    t0 = time.time()
    solve_for_kappa(kappa, N, Nt)
    gc.collect()
    print(f"  kappa={kappa:.3f} elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    with Pool(processes=num_workers) as pool:
        pool.map(_worker, kappa_list)
    print(f"\nTotal elapsed: {time.time() - start_time:.1f}s")
