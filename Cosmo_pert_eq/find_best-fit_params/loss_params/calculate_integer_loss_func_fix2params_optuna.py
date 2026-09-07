"""
Optuna optimization over pairs of cosmological parameters with the other two
fixed at the best-fit value.  Runs N_TRIALS trials per pair, one pair at a time.

Pairs (row, col) in order:
  0. OmegaM   - OmegaK
  1. OmegaM   - Omegab_ratio
  2. OmegaM   - h
  3. OmegaK   - Omegab_ratio
  4. OmegaK   - h
  5. Omegab_ratio - h

Results are written to:
  data_folder/fix2params/log_<pair_name>.txt   (per-pair)
  data_folder/fix2params/master_log_fix2params.txt  (combined, same format as master_log.txt)
"""

import os
import multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import sys
import optuna
import argparse
import classy
from classy import CosmoComputationError
custom_path = '../find_cosmo_params/'
sys.path.insert(0, custom_path) 
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK

# ── configuration ─────────────────────────────────────────────────────────────
nu_spacing4_bestfit = [0.4444730086581741, -0.03940583926181416, 0.16336760287147203, 0.5615471381223595, 2.106364,0.977438,0.059580] # best-fit parameters 

best_params = np.array(nu_spacing4_bestfit[:4])   # [OmegaM, OmegaK, Omegab_ratio, h]
param_names = ['OmegaM', 'OmegaK', 'Omegab_ratio', 'h']

N_ncdm = 1
m_ncdm = 0.06
Neff   = 3.046

N_TRIALS    = 500
n_processes = int(os.environ.get(
    "N_PROCESSES",
    os.environ.get("SLURM_CPUS_PER_TASK", "1")
))
n_processes = max(1, n_processes)

data_folder = './data/'
pair_folder = os.path.join(data_folder, 'fix2params')
os.makedirs(pair_folder, exist_ok=True)

combined_log = os.path.join(pair_folder, 'master_log_fix2params.txt')

# parameter ranges: same formula as plot_loss_param_space.py so they stay in sync
_data = []
with open(os.path.join(custom_path, 'data/master_log.txt')) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        _data.append([float(x) for x in parts[4:8]])
_params_all = np.array(_data)

p_ranges = []
for i in range(4):
    EXPAND = 1.3
    span = _params_all[:, i].max() - _params_all[:, i].min()
    p_ranges.append((best_params[i] - EXPAND * span/4, best_params[i] + EXPAND * span/4)) # slightly wider range than plot_loss_param_space.py

pbounds = {name: p_ranges[i] for i, name in enumerate(param_names)}

PAIRS = [
    (0, 1),  # OmegaM - OmegaK
    (0, 2),  # OmegaM - Omegab_ratio
    (0, 3),  # OmegaM - h
    (1, 2),  # OmegaK - Omegab_ratio
    (1, 3),  # OmegaK - h
    (2, 3),  # Omegab_ratio - h
]


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
        compute_U_matrices(params, z_rec, worker_folder, n_processes)
        compute_X_recs(params, z_rec, worker_folder)
        return compute_allowedK(params, worker_folder)
    except CosmoComputationError as e:
        print(f"  CLASS error for params {params}: {e}")
        return None


def _calculate_integer_loss(allowedK_integer, n_min=12, spacing_tol_factor=3,
                           w_slope=1.0, w_integer=0.0):
    """
    Loss for K sequences that asymptotically converge as k increases.
    Uses:
      1. A 2-point midpoint filter to kill alternating (+)/(-) wiggles.
      2. A 'plateau' inverted-parabola weighting to equally weight the entire tail,
         improving robustness against noise in the final points.
    """
    if allowedK_integer is None:
        return np.inf, np.inf, np.inf

    k = np.array(allowedK_integer)
    if len(k) < n_min + 3:
        return np.inf, np.inf, np.inf

    # --- 1. Robust trimming of non-physical jumps ---
    spacings = np.diff(k)
    nu_spacing = 4.0
    median_spacing = np.median(spacings)
    mad = np.median(np.abs(spacings - median_spacing))
    threshold = spacing_tol_factor * max(mad, 1e-4)

    end_idx = len(k)
    for i in range(len(spacings) - 1, len(spacings) // 2, -1):
        if abs(spacings[i] - median_spacing) > threshold:
            end_idx = i + 1

    k = k[:end_idx]
    n = len(k)
    if n < n_min:
        return np.inf, np.inf, np.inf

    # --- 2. 2-Point Midpoint Filter (Kills alternating wiggles) ---
    k_smooth = (k[:-1] + k[1:]) / 2.0
    indices_smooth = np.arange(len(k_smooth), dtype=float) + 0.5

    # --- 3. NEW: The User's "Plateau" Weighting (-x^2 style) ---
    # Create z from 0.0 to 1.0 representing position in the smooth array
    z_sp = np.linspace(0, 1, len(k_smooth) - 1)
    sp_weights = 1.0 - (1.0 - z_sp)**2 
    sp_weights /= sp_weights.sum()  # Normalize to sum to 1

    z_int = np.linspace(0, 1, len(k_smooth))
    int_weights = 1.0 - (1.0 - z_int)**2
    int_weights /= int_weights.sum() # Normalize to sum to 1

    # --- 4. Slope Loss ---
    spacings_smooth = np.diff(k_smooth)                         
    tail_slope_loss = np.sum(sp_weights * (spacings_smooth - nu_spacing) ** 2)

    # --- 5. Integer Loss ---
    weighted_mean_intercept = np.sum(int_weights * (k_smooth - nu_spacing * indices_smooth))
    ideal_intercept = np.round(weighted_mean_intercept)
    ideal_sequence  = ideal_intercept + nu_spacing * indices_smooth
    
    tail_integer_loss = np.sum(int_weights * (k_smooth - ideal_sequence) ** 2)

    total_loss = w_slope * tail_slope_loss + w_integer * tail_integer_loss
    return tail_slope_loss, tail_integer_loss, total_loss



# ── objective factory ─────────────────────────────────────────────────────────
def _make_objective(row, col, pair_log):
    def objective(trial):
        params = best_params.copy()
        params[col] = trial.suggest_float(param_names[col],
                                          *pbounds[param_names[col]])
        params[row] = trial.suggest_float(param_names[row],
                                          *pbounds[param_names[row]])

        worker_folder = os.path.join(
            pair_folder,
            f'worker_{os.getpid()}',
            f'pair_{row}_{col}_trial_{trial.number}'
        )
        os.makedirs(worker_folder, exist_ok=True)

        try:
            allowedK = _calculate_allowedK(params.tolist(), worker_folder)
            slope_loss, int_loss, total_loss = _calculate_integer_loss(allowedK)
        except (ValueError, CosmoComputationError):
            raise optuna.TrialPruned("CLASS error")

        if not np.isfinite(total_loss):
            raise optuna.TrialPruned("Loss is inf or NaN")

        k_str = ",".join(map(str, allowedK))
        line  = (f"{total_loss:.6e} {slope_loss:.6e} {int_loss:.6e} 0.00 "
                 + " ".join(map(str, params)) + f" {k_str}\n")

        with open(pair_log, 'a') as f:
            f.write(line)

        return total_loss
    return objective


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-index",
        type=int,
        default=None,
        help="Pair index 0..5. If omitted, run all pairs."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help="Target number of completed trials for this pair."
    )
    args = parser.parse_args()

    N_TRIALS = args.n_trials

    if args.pair_index is None:
        pair_indices = range(len(PAIRS))
    else:
        pair_indices = [args.pair_index]

    for pair_index in pair_indices:
        row, col = PAIRS[pair_index]
        name = f"{param_names[row]}_{param_names[col]}"
        print(f"\n{'='*60}")
        print(f"Pair index: {pair_index}")
        print(f"Pair: {name}  (fixing other two at best-fit)")
        print(f"  {param_names[row]} in {pbounds[param_names[row]]}")
        print(f"  {param_names[col]} in {pbounds[param_names[col]]}")
        print(f"  fixed: ", end='')
        for i, pn in enumerate(param_names):
            if i not in (row, col):
                print(f"{pn}={best_params[i]:.6f}", end='  ')
        print()

        pair_log = os.path.join(pair_folder, f'log_{name}.txt')
        db_url   = f"sqlite:///{os.path.join(pair_folder, f'study_{name}.db')}?timeout=60"

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=min(50, max(20, N_TRIALS // 3)),
            multivariate=True,
            seed=1234 + 10 * row + col
        )

        study = optuna.create_study(
            study_name=name,
            storage=db_url,
            direction='minimize',
            sampler=sampler,
            load_if_exists=True,
        )

        done = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE)
        remaining = max(0, N_TRIALS - done)

        if remaining == 0:
            print(f"  Already completed {done} trials — skipping.")
            continue

        print(f"  {done} trials already done, running {remaining} more...")
        study.optimize(_make_objective(row, col, pair_log), n_trials=remaining)

        try:
            print(f"  Best total_loss: {study.best_value:.6e}  "
                  f"at {study.best_params}")
        except ValueError:
            print("  No completed trials (all pruned).")

    print("\nAll pairs done.")
    print(f"Combined log: {combined_log}")
