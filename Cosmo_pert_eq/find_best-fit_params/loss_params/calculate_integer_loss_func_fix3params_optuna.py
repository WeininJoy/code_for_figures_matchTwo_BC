"""
Optuna optimization over each cosmological parameter individually, with the
other three fixed at best-fit values.  Runs N_TRIALS trials per parameter.

Parameters (param_index):
  0. OmegaM
  1. OmegaK
  2. Omegab_ratio
  3. h

Results are written to:
  data_folder/fix3params/log_<param_name>.txt        (per-parameter)
  data_folder/fix3params/master_log_fix3params.txt    (combined, same format as master_log.txt)
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

N_TRIALS    = 100
n_processes = int(os.environ.get(
    "N_PROCESSES",
    os.environ.get("SLURM_CPUS_PER_TASK", "1")
))
n_processes = max(1, n_processes)

data_folder  = './data/'
param_folder = os.path.join(data_folder, 'fix3params')
os.makedirs(param_folder, exist_ok=True)

combined_log = os.path.join(param_folder, 'master_log_fix3params.txt')

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
    p_ranges.append((best_params[i] - EXPAND * span / 4,
                     best_params[i] + EXPAND * span / 4))

pbounds = {name: p_ranges[i] for i, name in enumerate(param_names)}


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
    if allowedK_integer is None:
        return np.inf, np.inf, np.inf

    k = np.array(allowedK_integer)
    if len(k) < n_min + 3:
        return np.inf, np.inf, np.inf

    spacings       = np.diff(k)
    nu_spacing     = 4.0
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
    tail_slope_loss  = np.sum(sp_weights * (spacings_smooth - nu_spacing) ** 2)

    weighted_mean_intercept = np.sum(int_weights * (k_smooth - nu_spacing * indices_smooth))
    ideal_intercept         = np.round(weighted_mean_intercept)
    ideal_sequence          = ideal_intercept + nu_spacing * indices_smooth
    tail_integer_loss       = np.sum(int_weights * (k_smooth - ideal_sequence) ** 2)

    total_loss = w_slope * tail_slope_loss + w_integer * tail_integer_loss
    return tail_slope_loss, tail_integer_loss, total_loss


# ── objective factory ─────────────────────────────────────────────────────────
def _make_objective(param_idx, param_log):
    def objective(trial):
        params            = best_params.copy()
        params[param_idx] = trial.suggest_float(param_names[param_idx],
                                                *pbounds[param_names[param_idx]])

        worker_folder = os.path.join(
            param_folder,
            f'worker_{os.getpid()}',
            f'param_{param_idx}_trial_{trial.number}'
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

        with open(param_log, 'a') as f:
            f.write(line)
        with open(combined_log, 'a') as f:
            f.write(line)

        return total_loss
    return objective


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param-index",
        type=int,
        default=None,
        help="Parameter index 0..3. If omitted, run all parameters."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help="Target number of completed trials for this parameter."
    )
    args = parser.parse_args()

    N_TRIALS = args.n_trials

    if args.param_index is None:
        param_indices = range(4)
    else:
        param_indices = [args.param_index]

    for param_idx in param_indices:
        name = param_names[param_idx]
        print(f"\n{'='*60}")
        print(f"Parameter index: {param_idx}")
        print(f"Varying: {name}  in {pbounds[name]}")
        print(f"Fixed at best-fit: ", end='')
        for i, pn in enumerate(param_names):
            if i != param_idx:
                print(f"{pn}={best_params[i]:.6f}", end='  ')
        print()

        param_log = os.path.join(param_folder, f'log_{name}.txt')
        db_url    = f"sqlite:///{os.path.join(param_folder, f'study_{name}.db')}?timeout=60"

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=min(50, max(20, N_TRIALS // 3)),
            multivariate=False,   # 1-D scan: univariate TPE is sufficient
            seed=1234 + param_idx
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
        study.optimize(_make_objective(param_idx, param_log), n_trials=remaining)

        try:
            print(f"  Best total_loss: {study.best_value:.6e}  "
                  f"at {study.best_params}")
        except ValueError:
            print("  No completed trials (all pruned).")

    print("\nAll parameters done.")
    print(f"Combined log: {combined_log}")
