"""
Generate data for allowedK and integerK calculations using cosmological parameters.
"""
import numpy as np
import os
import time
import shutil
from scipy.optimize import root_scalar
import classy
from classy import CosmoComputationError
from Higher_Order_Finding_U_Matrices import compute_U_matrices
from Higher_Order_Finding_Xrecs import compute_X_recs
from Higher_Order_Solving_for_Vrinf import compute_allowedK
from Higher_Order_Finding_U_Matrices_TimeSeries_parallel import compute_U_matrices_timeseries

nu_spacing = 4
print("nu_spacing = ", nu_spacing)

# Detect number of CPUs
n_processes = 4
print(f"Using {n_processes} parallel processes")

#################
# Parameters
#################
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
N_ncdm = 1
m_ncdm = 0.06
epsilon = 1e-2
cosmo_param_bool = True 
nu_spacing4_bestfit = [0.4444730086581741, -0.03940583926181416, 0.16336760287147203, 0.5615471381223595, 2.106364,0.977438,0.059580] # best-fit cosmological parameters

##################
def cosmological_parameters(mt, kt, h):
    Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

    def solve_a0(Omega_r, rt, mt, kt):
        def f(a0):
            return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
        sol = root_scalar(f, bracket=[1, 1.e3])
        return sol.root

    a0 = solve_a0(Omega_r, rt, mt, kt)
    s0 = 1/a0
    Omega_lambda = Omega_r * a0**4
    Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
    Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
    return s0, Omega_lambda, Omega_m, Omega_K

##################
def calculate_z_rec(params):
    if cosmo_param_bool==True:
        OmegaM, OmegaK, omega_b_ratio, h = params
    else:
        mt, kt, omega_b_ratio, h = params
        s0, Omega_lambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
        

    CMB_params = {
        'output': 'tCl',
        'h': h,
        'Omega_b': omega_b_ratio*OmegaM,
        'Omega_cdm': (1.-omega_b_ratio) *OmegaM,
        'Omega_k': float(OmegaK),
        'A_s': nu_spacing4_bestfit[4]*1e-9,
        'n_s': nu_spacing4_bestfit[5],
        'N_ncdm': N_ncdm,
        'm_ncdm': m_ncdm,
        'N_ur': Neff - N_ncdm,
        'tau_reio': nu_spacing4_bestfit[6],
        'lensing': 'no'
    }

    cosmo = classy.Class()
    cosmo.set(CMB_params)
    cosmo.compute()

    derived = cosmo.get_current_derived_parameters(['z_rec'])
    z_rec = derived['z_rec']

    cosmo.struct_cleanup()
    cosmo.empty()

    return z_rec

def calculate_allowedK(params, z_rec, kvalues , folder_path, num_variables=200, cosmo_param_bool=False):
    compute_U_matrices(params, z_rec, kvalues, folder_path, num_variables, n_processes, cosmo_param_bool=cosmo_param_bool)
    compute_X_recs(params, z_rec, folder_path, num_variables, cosmo_param_bool=cosmo_param_bool)
    compute_allowedK(params, folder_path, num_variables, cosmo_param_bool=cosmo_param_bool)

def calculate_Umatrices_Xrec(params, z_rec, kvalues, folder_path,
                              num_variables=200, cosmo_param_bool=False):
    compute_U_matrices(params, z_rec, kvalues, folder_path, num_variables, n_processes, cosmo_param_bool=cosmo_param_bool)
    compute_X_recs(params, z_rec, folder_path, num_variables, cosmo_param_bool=cosmo_param_bool)


def pad_to_max_nv(arrays, axis=1):
    """Zero-pad arrays along `axis` so all have the same size as the largest."""
    max_size = max(a.shape[axis] for a in arrays)
    padded = []
    for a in arrays:
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (0, max_size - a.shape[axis])
        padded.append(np.pad(a, pad_width))
    return padded


def merge_data_chunks(chunk_folders, output_folder):
    """
    Concatenate static U-matrix data from per-range chunk folders along the
    k-axis (axis 0) into a single output folder.

    Chunks may have different num_variables (larger for high-k).  Arrays with a
    num_variables axis (ABCmatrices, DEFmatrices, GHIvectors) are zero-padded
    along that axis to the maximum size before concatenation.
    X1matrices and X2matrices have shape (N_k, 6, 4) and need no padding.
    """
    os.makedirs(output_folder, exist_ok=True)
    # Files whose axis-1 is num_variables and must be padded
    nv_files = {'L70_ABCmatrices', 'L70_DEFmatrices', 'L70_GHIvectors'}
    all_files = ['L70_kvalues', 'L70_ABCmatrices', 'L70_DEFmatrices',
                 'L70_GHIvectors', 'L70_X1matrices', 'L70_X2matrices',
                 'L70_recValues']
    for fname in all_files:
        arrays = [np.load(os.path.join(f, fname + '.npy'))
                  for f in chunk_folders
                  if os.path.exists(os.path.join(f, fname + '.npy'))]
        if not arrays:
            continue
        if fname in nv_files:
            arrays = pad_to_max_nv(arrays, axis=1)
        np.save(os.path.join(output_folder, fname), np.concatenate(arrays, axis=0))
    print(f"Merged {len(chunk_folders)} static-data chunks → {output_folder}")


def merge_timeseries_chunks(chunk_folders, output_folder):
    """
    Concatenate timeseries data from per-range chunk folders along the k-axis
    (axis 0) into a single output folder.

    All chunks must share the same t_grid — ensured by passing
    k_for_endtime=global_k_min to compute_U_matrices_timeseries for every chunk.
    The t_grid from the first chunk is used as the common grid.

    Chunks may have different num_variables; timeseries arrays are zero-padded
    along axis 1 (the num_variables axis) before concatenation.
    """
    os.makedirs(output_folder, exist_ok=True)
    t_grid = np.load(os.path.join(chunk_folders[0], 't_grid.npy'))
    np.save(os.path.join(output_folder, 't_grid.npy'), t_grid)

    # Files whose axis-1 is num_variables and must be padded
    nv_files = {'L70_ABC_solutions', 'L70_DEF_solutions', 'L70_GHI_solutions'}
    all_files = ['L70_kvalues', 'L70_ABC_solutions',
                 'L70_DEF_solutions', 'L70_GHI_solutions']
    for fname in all_files:
        arrays = [np.load(os.path.join(f, fname + '.npy'))
                  for f in chunk_folders
                  if os.path.exists(os.path.join(f, fname + '.npy'))]
        if not arrays:
            continue
        if fname in nv_files:
            arrays = pad_to_max_nv(arrays, axis=1)
        np.save(os.path.join(output_folder, fname), np.concatenate(arrays, axis=0))
    print(f"Merged {len(chunk_folders)} timeseries chunks → {output_folder}")

def calculate_integer_loss(allowedK_integer, n_min=12, spacing_tol_factor=3,
                           w_slope=1.0, w_integer=0.0):
                            # w_slope=20.0, w_integer=1.0):
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


######################################
# Main Execution
######################################
params = nu_spacing4_bestfit[:4] # [mt, kt, omega_b_ratio, h] or [Omega_m, Omega_K, omega_b_ratio, h] if cosmo_param_bool=True
discreteK_type = ['integerK', 'allowedK']
folder =  './data/'
z_rec = calculate_z_rec(params)

########### calculate allowedK ##############
kvalues_all_k50_65 = np.linspace(50,65,num=150); # data_all_k50-65, num_variables = 200
kvalues_all_k30_50 = np.linspace(30,50,num=200); # data_all_k30-50, num_variables = 160
kvalues_all_k0_30 = np.linspace(1e-5,30,num=300); # data_all_k0-30, num_variables = 120
kvalues_small_k = np.linspace(1e-5,3,num=100); # data_small_k, num_variables = 75
calculate_allowedK(params, z_rec, kvalues=kvalues_all_k50_65, folder_path=folder+'data_all_k50-65/', num_variables=200, cosmo_param_bool=cosmo_param_bool)
calculate_allowedK(params, z_rec, kvalues=kvalues_all_k30_50, folder_path=folder+'data_all_k30-50/', num_variables=160, cosmo_param_bool=cosmo_param_bool)
calculate_allowedK(params, z_rec, kvalues=kvalues_all_k0_30, folder_path=folder+'data_all_k0-30/', num_variables=120, cosmo_param_bool=cosmo_param_bool)
calculate_allowedK(params, z_rec, kvalues=kvalues_small_k, folder_path=folder+'data_small_k/', num_variables=75, cosmo_param_bool=cosmo_param_bool)
print("AllowedK calculation completed.")

########### generate discreteK data ##############

## kvalues based on allowedK
small_allowedK = np.load(folder + 'data_small_k/allowedK.npy')
allowedK_path = folder + 'data_all_k0-30/allowedK.npy'
kvalues_allowedK = np.load(allowedK_path); # only caluclate for the allowed K values
kvalues_allowedK = kvalues_allowedK[kvalues_allowedK > small_allowedK[-1] + 0.5* np.diff(kvalues_allowedK).mean()]  # remove k values that are already in small_allowedK
kvalues_allowedK = np.concatenate((small_allowedK[-3:], kvalues_allowedK))  # add a few small K values for better resolution at low k
kvalues_allowedK_k30_50 = np.load(folder + 'data_all_k30-50/allowedK.npy');
kvalues_allowedK_k50_65 = np.load(folder + 'data_all_k50-65/allowedK.npy');
kvalues_allowedK = np.concatenate((kvalues_allowedK, kvalues_allowedK_k30_50, kvalues_allowedK_k50_65))
print('kvalues_allowedK:', kvalues_allowedK)

## kvalues based on allowedK_integer
small_allowedK_integer = np.load(folder + 'data_small_k/allowedK_integer.npy')
allowedK_integer_path = folder + 'data_all_k0-30/allowedK_integer.npy'
allowedK_integer = np.load(allowedK_integer_path);
allowedK_integer = allowedK_integer[allowedK_integer > small_allowedK_integer[-1] + 0.5* np.diff(allowedK_integer).mean()]  # remove k values that are already in small_allowedK
allowedK_integer = np.concatenate((small_allowedK_integer[-3:], allowedK_integer))  # add a few small K values for better resolution at low k
allowedK_integer_k30_50 = np.load(folder + 'data_all_k30-50/allowedK_integer.npy');
allowedK_integer_k50_65 = np.load(folder + 'data_all_k50-65/allowedK_integer.npy');
allowedK_integer = np.concatenate((allowedK_integer, allowedK_integer_k30_50, allowedK_integer_k50_65))
print('allowedK_integer:', allowedK_integer)

n_ignore = len(allowedK_integer)-20
high_k = np.array(allowedK_integer[n_ignore:])
indices = np.arange(len(high_k))
slope, intercept = np.polyfit(indices, high_k, 1)
ideal_spacing = np.round(slope)
if ideal_spacing != nu_spacing: print("Warning: ideal spacing does not match nu_spacing!")
ideal_intercept = np.mean(high_k - ideal_spacing * indices)
ideal_k_sequence = ideal_spacing * indices + np.round(ideal_intercept)
small_integer_k_sequence = [ideal_k_sequence[0]-nu_spacing*i for i in range(1, n_ignore+1)][::-1]
whole_integer_k_sequence = np.concatenate((small_integer_k_sequence, ideal_k_sequence))
print("Ideal integer k sequence (after fitting):", whole_integer_k_sequence)
if cosmo_param_bool == True: 
    OmegaM, OmegaK, omega_b_ratio, h = params
    OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
else:
    mt, kt, omega_b_ratio, h = params
    s0, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h) 
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
a0=1; K=-OmegaK * a0**2 * H0**2
kvalues_integerK = [k * np.sqrt(K) for k in whole_integer_k_sequence]
print('kvalues_integerK:', kvalues_integerK)

# Shift kvalues_allowedK so that its high-k tail aligns with kvalues_integerK.
# At high k the two sequences share the same spacing but may differ by a constant
# offset.  We estimate that offset from the last few high-k allowedK points by
# matching each one to its nearest integerK neighbour, then shift uniformly.
_integerK_arr = np.array(kvalues_integerK)
_n_high = 20  # number of high-k points to use for the offset estimate
_high_allowed = kvalues_allowedK[-_n_high:]
_offsets = np.array([
    _integerK_arr[np.argmin(np.abs(_integerK_arr - ka))] - ka
    for ka in _high_allowed
])
_shift = np.mean(_offsets)
print(f"\nHigh-k per-point offsets (nearest integerK − allowedK): {_offsets}")
print(f"Mean shift applied to kvalues_allowedK: {_shift:.6f}")
kvalues_allowedK = kvalues_allowedK + _shift

# define integerK values as integer closest to the allowedK values
kvalues_integerK = [np.round(k/np.sqrt(K))*np.sqrt(K) for k in kvalues_allowedK]

# ---------------------------------------------------------------------------
# Per-k-range computation with appropriate num_variables for ODE accuracy.
# Full arrays (no truncation) are saved; the merge helpers zero-pad smaller
# arrays to the maximum num_variables size before concatenating.

# num_variables guidelines (increase with k for better multipole convergence):
#   small_k  (k <= 3)   → 75  is sufficient
#   k0_30    (k <= 30)  → 120
#   k30_50   (k <= 50)  → 150
#   k50_65   (k <= 65)  → 200
# ---------------------------------------------------------------------------

# Split allowedK values by k-range (they are already ordered)
kvals_allowedK_small = kvalues_allowedK[kvalues_allowedK <= 3]
kvals_allowedK_0_30  = kvalues_allowedK[(kvalues_allowedK > 3)  & (kvalues_allowedK <= 30)]
kvals_allowedK_30_50 = kvalues_allowedK[(kvalues_allowedK > 30) & (kvalues_allowedK <= 50)]
kvals_allowedK_50_65 = kvalues_allowedK[kvalues_allowedK > 50]

# Split integerK values by k-range
kvalues_integerK = np.array(kvalues_integerK)
kvals_integerK_small = kvalues_integerK[kvalues_integerK <= 3]
kvals_integerK_0_30  = kvalues_integerK[(kvalues_integerK > 3)  & (kvalues_integerK <= 30)]
kvals_integerK_30_50 = kvalues_integerK[(kvalues_integerK > 30) & (kvalues_integerK <= 50)]
kvals_integerK_50_65 = kvalues_integerK[kvalues_integerK > 50]

# Global k_min used to keep t_grid consistent across all timeseries chunks
k_min_global_allowedK = float(np.min(kvalues_allowedK))
k_min_global_integerK = float(np.min(kvalues_integerK))

# num_variables per k-range: larger values give better multipole accuracy at high k
range_nv = [75, 120, 160, 200]

# --- allowedK: static matrices + Xrec ---
allowedK_chunk_configs = [
    (kvals_allowedK_small, folder + 'data_allowedK_chunks/small_k/'),
    (kvals_allowedK_0_30,  folder + 'data_allowedK_chunks/k0_30/'),
    (kvals_allowedK_30_50, folder + 'data_allowedK_chunks/k30_50/'),
    (kvals_allowedK_50_65, folder + 'data_allowedK_chunks/k50_65/'),
]
for (kvals, chunk_folder), nv in zip(allowedK_chunk_configs, range_nv):
    if len(kvals) > 0:
        calculate_Umatrices_Xrec(params, z_rec, kvals, chunk_folder,
                                  num_variables=nv, cosmo_param_bool=cosmo_param_bool)

# --- integerK: static matrices + Xrec ---
integerK_chunk_configs = [
    (kvals_integerK_small, folder + 'data_integerK_chunks/small_k/'),
    (kvals_integerK_0_30,  folder + 'data_integerK_chunks/k0_30/'),
    (kvals_integerK_30_50, folder + 'data_integerK_chunks/k30_50/'),
    (kvals_integerK_50_65, folder + 'data_integerK_chunks/k50_65/'),
]
for (kvals, chunk_folder), nv in zip(integerK_chunk_configs, range_nv):
    if len(kvals) > 0:
        calculate_Umatrices_Xrec(params, z_rec, kvals, chunk_folder,
                                  num_variables=nv, cosmo_param_bool=cosmo_param_bool)

# Merge chunks into the final unified folders
allowedK_static_chunks = [f for (_, f) in allowedK_chunk_configs
                           if os.path.exists(os.path.join(f, 'L70_kvalues.npy'))]
integerK_static_chunks = [f for (_, f) in integerK_chunk_configs
                           if os.path.exists(os.path.join(f, 'L70_kvalues.npy'))]
merge_data_chunks(allowedK_static_chunks, folder + 'data_allowedK/')
merge_data_chunks(integerK_static_chunks, folder + 'data_integerK/')
print("U matrices and Xrec calculation completed.")


########### generate TimeSeries U matrices ##############
# --- allowedK timeseries ---
allowedK_ts_chunk_configs = [
    (kvals_allowedK_small, folder + 'data_allowedK_chunks/small_k_timeseries/'),
    (kvals_allowedK_0_30,  folder + 'data_allowedK_chunks/k0_30_timeseries/'),
    (kvals_allowedK_30_50, folder + 'data_allowedK_chunks/k30_50_timeseries/'),
    (kvals_allowedK_50_65, folder + 'data_allowedK_chunks/k50_65_timeseries/'),
]
for (kvals, chunk_folder), nv in zip(allowedK_ts_chunk_configs, range_nv):
    if len(kvals) > 0:
        compute_U_matrices_timeseries(params, z_rec, kvals, chunk_folder,n_processes,
                                      num_variables=nv,
                                      k_for_endtime=k_min_global_allowedK, cosmo_param_bool=cosmo_param_bool)

# --- integerK timeseries ---
integerK_ts_chunk_configs = [
    (kvals_integerK_small, folder + 'data_integerK_chunks/small_k_timeseries/'),
    (kvals_integerK_0_30,  folder + 'data_integerK_chunks/k0_30_timeseries/'),
    (kvals_integerK_30_50, folder + 'data_integerK_chunks/k30_50_timeseries/'),
    (kvals_integerK_50_65, folder + 'data_integerK_chunks/k50_65_timeseries/'),
]
for (kvals, chunk_folder), nv in zip(integerK_ts_chunk_configs, range_nv):
    if len(kvals) > 0:
        compute_U_matrices_timeseries(params, z_rec, kvals, chunk_folder,n_processes,
                                      num_variables=nv,
                                      k_for_endtime=k_min_global_integerK, cosmo_param_bool=cosmo_param_bool)

# Merge timeseries chunks
allowedK_ts_chunks = [f for (_, f) in allowedK_ts_chunk_configs
                       if os.path.exists(os.path.join(f, 't_grid.npy'))]
integerK_ts_chunks = [f for (_, f) in integerK_ts_chunk_configs
                       if os.path.exists(os.path.join(f, 't_grid.npy'))]
merge_timeseries_chunks(allowedK_ts_chunks, folder + 'data_allowedK_timeseries/')
merge_timeseries_chunks(integerK_ts_chunks, folder + 'data_integerK_timeseries/')
print("TimeSeries U matrices calculation completed.")
print("All data generation completed.")