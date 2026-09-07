######### Goal: Find best-fit parameter set by minimizing chi_eff_sq -> most simplier to observation
######### Method: use scipy.optimize.minimize() function
import clik
import numpy as np
import sys
import optuna
import os
import re
import pickle
# --- ADD THIS PATCH TO FIX NUMPY 2.0 PICKLE MISMATCH ---
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = sys.modules['numpy.core']
if 'numpy._core.multiarray' not in sys.modules:
    sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']
import time
import classy
print(classy.__file__)
# data is installed from clik
data = os.path.join(os.getcwd(),'/home/wnd22/rds/hpc-work/Polychord_solveb/clik_installs/data/planck_2018/baseline/plc_3.0') 

#################
# Parameters
#################
nu_spacing = 4
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
N_ncdm = 1
m_ncdm = 0.06
epsilon = 1e-2
k_cons = 0.05  # Mpc^-1, pivot scale for power spectrum
cosmo_param_bool = True
nu_spacing4_bestfit = [0.4444730086581741, -0.03940583926181416, 0.16336760287147203, 0.5615471381223595, 2.149037570071961, 0.9643873518055414, 0.06402059001822309] # best-fit cosmological parameters
## folders and files
name_str = 'cosmo_param_direc'
data_dir = f'./data'
cca_file = f"./cca_2D_dk_penalty_results_{name_str}.pickle"

# Write to the local RAM disk instead of the network drive
dat_name = f'/dev/shm/PPS_{os.getpid()}.dat'
file_root = f'{data_dir}/find_bestfit.txt'
# Persistent SQLite DB so Optuna continues from previous runs
DB_URL = f"sqlite:///{data_dir}/optuna_bestfit_As_ns_tau.db?timeout=60"
all_samples_file = f'{data_dir}/all_samples_As_ns_tau.txt'
start_params = nu_spacing4_bestfit[4:7]  # [As, ns, tau] — SQLite handles continuation

class PlanckLikelihood(object):
    """Baseline Planck Likelihood"""
    # Use TTTEEE, without lensing
    
    def __init__(self):
        self.plik = clik.clik(os.path.join(data, "hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"))
        self.lowl = clik.clik(os.path.join(data, "low_l/commander/commander_dx12_v3_2_29.clik"))
        self.lowE = clik.clik(os.path.join(data, "low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"))
    
    def __call__(self, cls, nuis):
        lkl = []
        for like in [self.plik, self.lowl, self.lowE]:
            #        for like in [self.plik]:
            lmaxes = like.get_lmax()
            dat = []
            order = ['tt','ee','bb','te','tb','eb']
            
            # print(order,len(lmaxes),len(order))
            for spec, lmax in zip(order, lmaxes):
                if lmax>-1:
                    if spec == 'pp':
                        dat += list(cls[spec][:lmax+1])
                    else:
                        dat += list(cls[spec][:lmax+1]* (1e6 * 2.7255)**2 )
        
            for param in like.get_extra_parameter_names():
                dat.append(nuis[param])
    
            lkl.append(like(dat))
    
        return np.array(lkl).flatten()


# calculate and output PPS.dat file with parameters = [As, ns, tau]
 
def PPS_dat(params, eigenvecs_sparse, k_values, dat_name):
    """
    Computes the Observed Power Spectrum based on the Quasi-Fourier Basis.
    
    Physics Logic:
    1. We assume the universe excites the 'Quasi-Fourier' modes (eigenvecs_sparse)
       independently.
    2. The amplitude of each mode is determined by its dominant scale (k_dom)
       following the standard inflation power law.
    3. We sum the contributions of all Quasi-Fourier modes to find the 
       total power observed at a specific wavenumber k.
       
    Args:
        eigenvecs_sparse: (N_k x N_modes) matrix. Columns are physical modes Psi_lambda.
                          Rows are standard Fourier wavenumbers k.
        k_values: Array of k values corresponding to the rows of the matrix.
        As: Scalar amplitude of fluctuations.
        ns: Spectral index.
        
    Returns:
        P_obs: The observed power spectrum at each k.
        P_std: The standard power law prediction for comparison.
        k_dominants: The dominant k identified for each eigenvector.
    """
    As, ns, tau = params[0]*1e-9, params[1], params[2]
    N_k, N_modes = eigenvecs_sparse.shape
    
    # Placeholder for the observed power
    P_obs = np.zeros(len(k_values)) # np.zeros(N_k)
    
    # 1. Identify the dominant k for each sparse eigenvector
    # This labels the mode: "This is the physical manifestation of k=5"
    k_dominants = []
    
    # Pre-calculate the Primordial Power assigned to each Eigenvector
    # P_eigenvector[lambda] = As * (k_dom)^(ns-1)
    P_assigned_to_modes = np.zeros(N_modes)
    
    for lam in range(N_modes):
        # Get the vector Psi_lambda
        vec = eigenvecs_sparse[:, lam]
        
        # Find index of max amplitude
        dom_idx = np.argmax(np.abs(vec))
        k_dom = k_values[dom_idx]
        k_dominants.append(k_dom)
        
        # Assign Power based on this dominant scale
        # This is the "Adiabatic Vacuum" assumption
        P_assigned_to_modes[lam] = As * (k_dom / k_cons)**(ns - 1)

    # 2. Calculate Observed Power at each Fourier wavenumber k
    # An observer measuring mode k sees contributions from ALL physical modes
    # that have a non-zero projection onto k.
    
    for i, k_val in enumerate(k_values):
        # The row i of eigenvecs_sparse tells us how much each physical mode lambda
        # overlaps with Fourier mode k.
        for lam in range(N_modes):
            projection_weights = np.abs(eigenvecs_sparse[i, lam])**2
            # Sum of (Weight * Power_of_that_mode)
            # P_obs(k) = Sum_lambda |U_{k,lambda}|^2 * P_prim(k_dom_lambda)
            P_obs[i] += projection_weights * P_assigned_to_modes[lam]
      
    def P_std(k): 
        return As * (k / k_cons)**(ns - 1)

    # ── Truncate the quick fluctuating region in high-k (caused by numerical artifacts) ───────────────────────────────
    window = 5
    trunc_idx = len(k_physical) - 1  # Default if the condition is somehow not met

    # Scan BACKWARDS from the highest k towards the small k
    for i in range(len(k_physical) - 1, window - 1, -1):
        
        # Take the moving average of the 5 points ending at current index i
        window_P = P_obs[i - window + 1 : i + 1]
        local_avg = np.mean(window_P)
        std_val = P_std(k_physical[i])
        
        # When moving backward, the noise average stays around 1.0 * P_std.
        # Once it drops below 0.8 * P_std, we have hit the physical dip at k ~ 8e-3!
        if local_avg < 0.8 * std_val:
            
            # To ensure CLASS cubic splines don't overshoot, we shouldn't cut at the 
            # bottom of the dip. We look slightly to the right (just as it exits the dip) 
            # to find where P_observed exactly crosses P_std.
            search_start = i
            search_end = min(len(k_physical), i + 6)
            
            best_local = np.argmin(np.abs(
                P_obs[search_start:search_end] - 
                P_std(k_physical[search_start:search_end])
            ))
            
            # Set the truncate point exactly at the crossing
            trunc_idx = search_start + best_local
            break

    print(f"Truncating right after the physical dip at index {trunc_idx}, k = {k_physical[trunc_idx]:.4e}")

    k_clean = k_physical[:trunc_idx + 1]
    P_clean = P_obs[:trunc_idx + 1]

    # ── ASSEMBLE FULL ARRAYS WITH PADDING ──────────────────────────────────
    
    # Standard low-k padding
    k_pad_lo = np.array([0.9e-6, 1.e-6]) # Mpc^-1
    P_pad_lo = np.full(2, 1e-300)

    # High-k padding appended after the truncate point
    k_pad_hi = np.geomspace(k_clean[-1] * 1.01, 10000.0, num=100)
    P_pad_hi = P_std(k_pad_hi)

    k_full = np.concatenate([k_pad_lo, k_clean, k_pad_hi])
    P_full = np.concatenate([P_pad_lo, P_clean, P_pad_hi])

    header = 'k  P_s(k)  (external_Pk format)'
    np.savetxt(dat_name,
            np.column_stack([k_full, P_full]),
            fmt='%.8e %.8e', header=header)
    
    return 0


# Runs the CLASS code
def run_class(params, eigenvecs_sparse, k_values, dat_name):
    
    # calulate PPS and make dat file
    PPS_dat(params, eigenvecs_sparse, k_values, dat_name)
   
    # parameters
    As, ns, tau = params
    
    ####### solve_b cls
    OmegaM, OmegaK, omega_b_ratio, h = nu_spacing4_bestfit[:4]
    params_pstb = {
        'omega_b': OmegaM * omega_b_ratio * h**2,
        'omega_cdm': (1-omega_b_ratio)* OmegaM * h**2,
        'h': h,
        'tau_reio': tau,
        'Omega_k':  OmegaK,
        'N_ncdm': N_ncdm,
        'm_ncdm': m_ncdm,
        'N_ur': Neff - N_ncdm,
        'lensing': 'yes',
        'output': 'tCl pCl lCl mPk',
        'P_k_ini type': 'external_Pk',
        # 'P_k_max_1/Mpc':3.0,
        'l_max_scalars':2508,
        'command': 'cat ' + dat_name
    }

    pstb = classy.Class()
    pstb.set(params_pstb)
    pstb.compute()
    cls = pstb.lensed_cl(2508)
    pstb.struct_cleanup()
    pstb.empty()

    return cls


# Global variable for best chi^2
best_chisq = 2e+30 

# Computes TT spectrum and returns chi^2 for given set of parameters params in form  [logA_SR, N_star, log10f_i, omega_k, H0] using linear quantisation
def run_TT(params, eigenvecs_sparse, k_values, dat_name, filename, lkl):
        
        # try:
    # Find corresponding spectra
    cls = run_class(params, eigenvecs_sparse, k_values, dat_name)
    nuisance_params = [1.00044, 46.1, 0.66, 7.08, 248.2, 50.7, 53.3, 121.9, 0., 8.80, 11.01, 20.16, 95.5, 0.1138, 0.1346, 0.479, 0.225, 0.665, 2.082, 0.99974, 0.99819]

    nuis={
        'ycal':nuisance_params[0],
        'A_cib_217':nuisance_params[1],
        'xi_sz_cib':nuisance_params[2],
        'A_sz':nuisance_params[3],
        'ps_A_100_100':nuisance_params[4],
        'ps_A_143_143':nuisance_params[5],
        'ps_A_143_217':nuisance_params[6],
        'ps_A_217_217':nuisance_params[7],
        'ksz_norm':nuisance_params[8],
        'gal545_A_100':nuisance_params[9],
        'gal545_A_143':nuisance_params[10],
        'gal545_A_143_217':nuisance_params[11],
        'gal545_A_217':nuisance_params[12],
        'galf_TE_A_100':nuisance_params[13],
        'galf_TE_A_100_143':nuisance_params[14],
        'galf_TE_A_100_217':nuisance_params[15],
        'galf_TE_A_143':nuisance_params[16],
        'galf_TE_A_143_217':nuisance_params[17],
        'galf_TE_A_217':nuisance_params[18],
        'calib_100T':nuisance_params[19],
        'calib_217T':nuisance_params[20],
        'cib_index':-1.3, #no range given in table so assume fixed
        #-------------------------------------------------------------------
        # These are all set to 1, so assume that these are fixed -----------
        'A_cnoise_e2e_100_100_EE':1.,
        'A_cnoise_e2e_143_143_EE':1.,
        'A_cnoise_e2e_217_217_EE':1.,
        'A_sbpx_100_100_TT':1.,
        'A_sbpx_143_143_TT':1.,
        'A_sbpx_143_217_TT':1.,
        'A_sbpx_217_217_TT':1.,
        'A_sbpx_100_100_EE':1.,
        'A_sbpx_100_143_EE':1.,
        'A_sbpx_100_217_EE':1.,
        'A_sbpx_143_143_EE':1.,
        'A_sbpx_143_217_EE':1.,
        'A_sbpx_217_217_EE':1.,
        'A_pol':1,
        'A_planck':1.,
        #-------------------------------------------------------------------
        # These are fixed from Planck 2018 Likelihood Paper, Table 16 ------
        'galf_EE_A_100':0.055,
        'galf_EE_A_100_143':0.040,
        'galf_EE_A_100_217':0.094,
        'galf_EE_A_143':0.086,
        'galf_EE_A_143_217':0.21,
        'galf_EE_A_217':0.70,
        'calib_100P':1.021,
        'calib_143P':0.966,
        'calib_217P':1.04,
        #-------------------------------------------------------------------
        # These are fixed from Planck 2018 Likelihood Paper, pg 39 ---------
        'galf_EE_index':-2.4,
        'galf_TE_index':-2.4,
    #-------------------------------------------------------------------
    }

    plik, lowl, lowE = -2 * lkl(cls, nuis)
    chi_eff_sq = plik + lowl + lowE
    print('chi_eff_sq='+str(chi_eff_sq))

    # Update best_chisq
    global best_chisq
    if chi_eff_sq < best_chisq:
        best_chisq = chi_eff_sq
        with open(filename, 'w') as f:
            print(plik, lowl, lowE, chi_eff_sq, *params, 'False', file=f)

    return plik, lowl, lowE, chi_eff_sq


print("Initializing Planck Likelihood...")
lkl = PlanckLikelihood()
print("Planck Likelihood Initialized!")


def objective(trial):
    # Optuna natively handles your priors
    As = trial.suggest_float('As', 1.5, 3.5)
    ns = trial.suggest_float('ns', 0.9, 1.1)
    tau = trial.suggest_float('tau', 0.03, 0.08)
    params = [As, ns, tau]

    # Create a unique RAM-disk filename
    worker_pid = os.getpid()
    unique_dat_name = f'/dev/shm/PPS_{worker_pid}.dat'

    plik = lowl = lowE = chi_eff_sq = 2e+30
    try:
        plik, lowl, lowE, chi_eff_sq = run_TT(
            params, eigenvecs_sparse, k_physical, unique_dat_name, file_root, lkl
        )
    except Exception as e:
        print(f"Trial failed: {e}")
    finally:
        if os.path.exists(unique_dat_name):
            os.remove(unique_dat_name)

    # Log every trial (including failed ones) for post-analysis
    with open(all_samples_file, 'a') as f:
        f.write(f"{plik:.6e} {lowl:.6e} {lowE:.6e} {chi_eff_sq:.6e} {As:.8f} {ns:.8f} {tau:.8f}\n")

    return chi_eff_sq

def get_data_optuna():
    # Connect to (or create) the persistent SQLite study.
    # load_if_exists=True means subsequent runs continue where the previous left off.
    for attempt in range(10):
        try:
            study = optuna.create_study(
                study_name="bestfit_As_ns_tau",
                storage=DB_URL,
                direction="minimize",
                load_if_exists=True,
            )
            break
        except Exception as e:
            print(f"DB connection attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    else:
        raise RuntimeError("Could not connect to Optuna database after 10 attempts.")

    # Write the all-samples header only if the file is new
    if not os.path.exists(all_samples_file):
        with open(all_samples_file, 'w') as f:
            f.write("# plik lowl lowE chi_eff_sq As ns tau\n")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        # Fresh study — seed with start_params so the first trial isn't random
        study.enqueue_trial({'As': start_params[0], 'ns': start_params[1], 'tau': start_params[2]})
        print(f"Fresh study — seeding first trial from start_params: {start_params}")
    else:
        print(f"Resuming study: {len(completed)} completed trials found.")
        try:
            print(f"Current best: chi_eff_sq = {study.best_value:.4f} at {study.best_params}")
        except ValueError:
            print("No completed trials with a finite value yet.")

    print("Starting sequential Optuna optimization...")
    study.optimize(objective, n_trials=1000, n_jobs=1)

    best = study.best_params
    with open(file_root, 'w') as f:
        f.write(f"Best chi_eff_sq: {study.best_value}\n")
        f.write(f"{best['As']} {best['ns']} {best['tau']}\n")

    print(f"Optimization finished! Best chi_eff_sq: {study.best_value}")

if __name__ == '__main__':


    # Unpack parameters
    OmegaM, OmegaK, omega_b_ratio, h = nu_spacing4_bestfit[:4]
    OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    OmegaLambda = 1 - OmegaM - OmegaK - OmegaR

    # ── Load CCA sparse modes (A_sparse: N_k_int × m_cca) ───────────────────
    with open(cca_file, 'rb') as f:
        cca_results = pickle.load(f)
    A_sparse = cca_results['A_sparse']   # (N_k_int, m_cca)
    # Column-normalise: each mode vector becomes a unit vector (same as plot_heatmap)
    col_norms = np.linalg.norm(A_sparse, axis=0)
    col_norms = np.where(col_norms < 1e-30, 1.0, col_norms)
    A_sparse  = A_sparse / col_norms
    print(f"Loaded A_sparse (column-normalised): shape {A_sparse.shape}  "
          f"(N_k_int={A_sparse.shape[0]}, m_cca={A_sparse.shape[1]})")

    # ── Load k values ─────────────────────────────────────────────────────────
    k_vals_int = np.load(f'{data_dir}/integer_kvalues.npy')
    N_k_int = len(k_vals_int)
    m_cca   = A_sparse.shape[1]
    print(f"N_k_int={N_k_int}, m_cca={m_cca}")

    if A_sparse.shape[0] != N_k_int:
        raise ValueError(f"A_sparse rows ({A_sparse.shape[0]}) != N_k_int ({N_k_int})")

    eigenvecs_sparse = A_sparse
    print(f"eigenvecs shape: {eigenvecs_sparse.shape}")

    # ── Convert k to Mpc^{-1} ─────────────────────────────────────────────────
    # Unit: 1 [Lambda=c=1] = H0_phys * sqrt(3*OmegaLambda) [Mpc^{-1}]
    # Use comoving k (do NOT divide by a_present).

    c_kms = 299792.458                               # km/s
    H0_phys_Mpc_inv = (h * 100) / c_kms              # H0 in Mpc^{-1}
    conversion_factor = H0_phys_Mpc_inv * np.sqrt(3 * OmegaLambda)
    k_physical = k_vals_int * conversion_factor      # comoving k in Mpc^{-1}

    ### Run the optimization to find best-fit parameters
    get_data_optuna()
