"""
Compute integer_loss on a 10x10 grid for each pair of cosmological parameters,
with the other 2 fixed at best-fit values. Saves .npz files used by the
upper-triangle contour plots in plot_loss_param_space.py.
"""

import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import multiprocessing as mp
import multiprocessing.pool
import classy
from classy import CosmoComputationError

custom_path = '../find_cosmo_params/'
sys.path.insert(0, custom_path) 
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK

nu_spacing4_bestfit = [0.4444730086581741, -0.03940583926181416,  0.16112337683135397, 0.5615471381223595, 2.106364,0.977438,0.059580] # best-fit parameters
N_ncdm = 1
m_ncdm = 0.06
Neff   = 3.046

data_folder = './data/'
grid_folder = os.path.join(data_folder, 'grid_losses')
os.makedirs(grid_folder, exist_ok=True)

# ── parameter ranges: derived from existing data (same formula as plot script) ─
_data = []
with open(os.path.join(custom_path, 'data/master_log.txt')) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        _data.append([float(x) for x in parts[:8]])

_arr        = np.array(_data)
_params_all = _arr[:, 4:8]

best_params = np.array(nu_spacing4_bestfit[:4])
param_names = ['OmegaM', 'OmegaK', 'Omegab_ratio', 'h']

p_ranges = []
for i in range(4):
    span = _params_all[:, i].max() - _params_all[:, i].min()
    half = span / 4
    p_ranges.append((best_params[i] - half, best_params[i] + half))


# ── physics ───────────────────────────────────────────────────────────────────
def _calculate_z_rec(params):
    OmegaM, OmegaK, omega_b_ratio, h = params
    CMB_params = {
        'output': 'tCl',
        'h': h,
        'Omega_b': omega_b_ratio * OmegaM,
        'Omega_cdm': (1. - omega_b_ratio) * OmegaM,
        'Omega_k': float(OmegaK),
        'A_s': nu_spacing4_bestfit[4] * 1e-9,
        'n_s': nu_spacing4_bestfit[5],
        'N_ncdm': N_ncdm,
        'm_ncdm': m_ncdm,
        'N_ur': Neff - N_ncdm,
        'tau_reio': nu_spacing4_bestfit[6],
        'lensing': 'no',
    }
    cosmo = classy.Class()
    cosmo.set(CMB_params)
    cosmo.compute()
    z_rec = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
    cosmo.struct_cleanup()
    cosmo.empty()
    return z_rec


def _calculate_allowedK(params, worker_folder):
    try:
        z_rec = _calculate_z_rec(params)
        compute_U_matrices(params, z_rec, worker_folder, n_processes=1)
        compute_X_recs(params, z_rec, worker_folder)
        return compute_allowedK(params, worker_folder)
    except CosmoComputationError as e:
        print(f"  CLASS error for params {params}: {e}")
        return None


def _calculate_integer_loss(allowedK_integer, n_min=12, spacing_tol_factor=3,
                            w_slope=1.0, w_integer=0.0):
    if allowedK_integer is None:
        return np.inf, np.inf, np.inf

    k = np.array(allowedK_integer)
    if len(k) < n_min + 3:
        return np.inf, np.inf, np.inf

    spacings       = np.diff(k)
    nu_sp          = 4.0
    median_spacing = np.median(spacings)
    mad            = np.median(np.abs(spacings - median_spacing))
    threshold      = spacing_tol_factor * max(mad, 1e-4)

    end_idx = len(k)
    for i in range(len(spacings) - 1, len(spacings) // 2, -1):
        if abs(spacings[i] - median_spacing) > threshold:
            end_idx = i + 1

    k = k[:end_idx]
    if len(k) < n_min:
        return np.inf, np.inf, np.inf

    k_smooth       = (k[:-1] + k[1:]) / 2.0
    indices_smooth = np.arange(len(k_smooth), dtype=float) + 0.5

    z_sp       = np.linspace(0, 1, len(k_smooth) - 1)
    sp_weights = 1.0 - (1.0 - z_sp) ** 2
    sp_weights /= sp_weights.sum()

    z_int       = np.linspace(0, 1, len(k_smooth))
    int_weights = 1.0 - (1.0 - z_int) ** 2
    int_weights /= int_weights.sum()

    spacings_smooth  = np.diff(k_smooth)
    tail_slope_loss  = np.sum(sp_weights * (spacings_smooth - nu_sp) ** 2)

    weighted_mean_intercept = np.sum(int_weights * (k_smooth - nu_sp * indices_smooth))
    ideal_intercept         = np.round(weighted_mean_intercept)
    ideal_sequence          = ideal_intercept + nu_sp * indices_smooth
    tail_integer_loss       = np.sum(int_weights * (k_smooth - ideal_sequence) ** 2)

    total_loss = w_slope * tail_slope_loss + w_integer * tail_integer_loss
    return tail_slope_loss, tail_integer_loss, total_loss


# ── worker function ───────────────────────────────────────────────────────────
def _compute_grid_point(args):
    """Worker: compute losses for one grid point."""
    row, col, xi_val, yi_val, ji = args

    params      = best_params.copy()
    params[col] = xi_val   # x-axis in the plot
    params[row] = yi_val   # y-axis in the plot

    pid           = os.getpid()
    worker_folder = os.path.join(grid_folder, f'tmp_r{row}c{col}_pid{pid}', '')
    os.makedirs(worker_folder, exist_ok=True)

    allowedK = None
    try:
        allowedK                    = _calculate_allowedK(params.tolist(), worker_folder)
        slope_loss, int_loss, tloss = _calculate_integer_loss(allowedK)
    except Exception as e:
        print(f"  Error at ({row},{col}) ji={ji}: {e}")
        slope_loss, int_loss, tloss = np.inf, np.inf, np.inf

    j, i = ji
    print(f"[r={row},c={col}] ({j},{i}) slope_loss={slope_loss:.4e}  "
          f"({param_names[row]}={yi_val:.5f}, {param_names[col]}={xi_val:.5f})")
    # convert to array so it survives pickling (None stays None)
    k_arr = np.array(allowedK) if allowedK is not None else None
    return ji, slope_loss, int_loss, tloss, k_arr


# ── non-daemon pool (allows compute_U_matrices to spawn its own children) ─────
class _NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class _NoDaemonPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwargs):
        return _NoDaemonProcess(*args, **kwargs)


# ── main ──────────────────────────────────────────────────────────────────────
GRID_N = 10


if __name__ == '__main__':
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {n_workers} parallel workers  (grid size {GRID_N}x{GRID_N})\n")

    for row in range(4):
        for col in range(row + 1, 4):
            out_file = os.path.join(grid_folder, f'grid_r{row}_c{col}.npz')
            if os.path.exists(out_file):
                print(f"Skipping ({row},{col}) — already exists: {out_file}")
                continue

            xi = np.linspace(*p_ranges[col], GRID_N)
            yi = np.linspace(*p_ranges[row], GRID_N)

            # meshgrid convention: index [j, i] <-> (yi[j], xi[i])
            tasks = [
                (row, col, xi[i], yi[j], (j, i))
                for j in range(GRID_N)
                for i in range(GRID_N)
            ]

            print(f"=== Pair ({row},{col}): {param_names[row]} vs {param_names[col]}  "
                  f"({len(tasks)} points) ===")

            slope_loss_grid   = np.full((GRID_N, GRID_N), np.inf)
            integer_loss_grid = np.full((GRID_N, GRID_N), np.inf)
            total_loss_grid   = np.full((GRID_N, GRID_N), np.inf)
            # object array: each cell holds a 1-D float array (or None)
            allowedK_grid     = np.empty((GRID_N, GRID_N), dtype=object)

            with _NoDaemonPool(n_workers) as pool:
                for result in pool.imap_unordered(_compute_grid_point, tasks):
                    (j, i), sl, il, tl, k_arr = result
                    slope_loss_grid[j, i]   = sl
                    integer_loss_grid[j, i] = il
                    total_loss_grid[j, i]   = tl
                    allowedK_grid[j, i]     = k_arr

            np.savez(out_file,
                     xi=xi, yi=yi,
                     slope_loss=slope_loss_grid,
                     integer_loss=integer_loss_grid,
                     total_loss=total_loss_grid,
                     allowedK=allowedK_grid,
                     row=row, col=col)
            print(f"--> Saved: {out_file}\n")

    print("All pairs done.")
