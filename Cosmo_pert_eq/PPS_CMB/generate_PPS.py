import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import os

#################
# Parameters
#################
nu_spacing = 4
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5
Neff = 3.046
N_ncdm = 1
m_ncdm = 0.06
epsilon = 1e-2
k_cons = 0.05  # Mpc^-1, pivot scale for power spectrum 
cosmo_param_bool = True
nu_spacing4_bestfit = [0.4444730086581741, -0.03940583926181416, 0.16336760287147203, 0.5615471381223595, 2.180497037062433, 0.9672453294986536, 0.07235242742234273] # Best-fit parameters, including As,ns,tau
## folders and files
figures_dir = f'./figures'
cca_file = f"./linear_combine_sol/cca_2D_dk_penalty_results.pickle"

########################
# Calculate cosmological parameters based on the best-fit values
########################
if cosmo_param_bool==True:
    # Unpack parameters
    OmegaM, OmegaK, omega_b_ratio, h = nu_spacing4_bestfit[:4]
    OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
else:
    mt, kt, omega_b_ratio, h = nu_spacing4_bestfit[:4]

    def cosmological_parameters(mt, kt, h): 

        Omega_r = (1 + Neff*(7/8)*(4/11)**(4/3) ) * Omega_gamma_h2/h**2

        def solve_a0(Omega_r, rt, mt, kt):
            def f(a0):
                return a0**4 - 3*kt*a0**2 + mt*a0 + (rt-1./Omega_r)
            sol = root_scalar(f, bracket=[1, 1.e3])
            return sol.root

        a0 = solve_a0(Omega_r, rt, mt, kt)
        Omega_lambda = Omega_r * a0**4
        Omega_m = mt * Omega_lambda**(1/4) * Omega_r**(3/4)
        Omega_K = -3* kt * np.sqrt(Omega_lambda* Omega_r)
        return a0, Omega_lambda, Omega_m, Omega_K

    a_present, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)

H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
a0=1; K=-OmegaK * a0**2 * H0**2


def compute_observed_pps(eigenvecs_sparse, k_values, As=nu_spacing4_bestfit[4]*1e-9, ns=nu_spacing4_bestfit[5]):
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
      
    # Calculate Standard Law for comparison
    P_std = As * (k_values / k_cons)**(ns - 1)
    
    return P_obs, P_std, np.array(k_dominants)

# =============================================================================
# Execution & Plotting
# =============================================================================

if __name__ == "__main__":
    import pickle

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
    k_vals_int = np.load('../linear_combine_sol/data/data_integerK/L70_kvalues.npy')
    N_k_int = len(k_vals_int)
    m_cca   = A_sparse.shape[1]
    print(f"N_k_int={N_k_int}, m_cca={m_cca}")

    if A_sparse.shape[0] != N_k_int:
        raise ValueError(f"A_sparse rows ({A_sparse.shape[0]}) != N_k_int ({N_k_int})")

    eigenvecs = A_sparse
    print(f"eigenvecs shape: {eigenvecs.shape}")

    # ── Convert k to Mpc^{-1} ─────────────────────────────────────────────────
    # Unit: 1 [Lambda=c=1] = H0_phys * sqrt(3*OmegaLambda) [Mpc^{-1}]
    # Use comoving k (do NOT divide by a_present).

    c_kms = 299792.458                               # km/s
    H0_phys_Mpc_inv = (h * 100) / c_kms              # H0 in Mpc^{-1}
    conversion_factor = H0_phys_Mpc_inv * np.sqrt(3 * OmegaLambda)
    k_physical = k_vals_int * conversion_factor      # comoving k in Mpc^{-1}

    # ── Run the PPS calculation ───────────────────────────────────────────────
    P_observed, P_standard, k_doms = compute_observed_pps(eigenvecs, k_physical)

    # Standard PPS function
    As, ns = nu_spacing4_bestfit[4]*1e-9, nu_spacing4_bestfit[5]
    def P_std(k): 
        return As * (k / k_cons)**(ns - 1)

    # ── Truncate the quick fluctuating region in high-k (caused by numerical artifacts) ───────────────────────────────
    window = 5
    trunc_idx = len(k_physical) - 1  # Default if the condition is somehow not met

    # Scan BACKWARDS from the highest k towards the small k
    for i in range(len(k_physical) - 1, window - 1, -1):
        
        # Take the moving average of the 5 points ending at current index i
        window_P = P_observed[i - window + 1 : i + 1]
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
                P_observed[search_start:search_end] - 
                P_std(k_physical[search_start:search_end])
            ))
            
            # Set the truncate point exactly at the crossing
            trunc_idx = search_start + best_local
            break

    print(f"Truncating right after the physical dip at index {trunc_idx}, k = {k_physical[trunc_idx]:.4e}")

    k_clean = k_physical[:trunc_idx + 1]
    P_clean = P_observed[:trunc_idx + 1]

    # ── ASSEMBLE FULL ARRAYS WITH PADDING ──────────────────────────────────
    
    # Standard low-k padding
    k_pad_lo = np.array([0.9e-6, 1.e-6]) # Mpc^-1
    P_pad_lo = np.full(2, 1e-300)

    # High-k padding appended after the truncate point
    k_pad_hi = np.geomspace(k_clean[-1] * 1.01, 10000.0, num=100)
    P_pad_hi = P_std(k_pad_hi)

    k_full = np.concatenate([k_pad_lo, k_clean, k_pad_hi])
    P_full = np.concatenate([P_pad_lo, P_clean, P_pad_hi])

    dat_path = f'../PPS_CMB/_pps_external.dat'
    header = 'k  P_s(k)  (external_Pk format)'
    np.savetxt(dat_path,
            np.column_stack([k_full, P_full]),
            fmt='%.8e %.8e', header=header)
    print(f"Saved discrete PPS .dat to {dat_path}")


    # Create output directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)

    # --- Plot 1: The Power Spectra ---
    P_smooth_highK = [P_std(k) for k in k_physical[trunc_idx:]]
    P_observe_smooth_highK = np.concatenate([P_observed[:trunc_idx], P_smooth_highK])
    plt.figure(figsize=(8, 4.5))
    plt.plot(k_physical, np.log(1e10*P_standard), color='k', linewidth=2, label=r'Standard $\Lambda$CDM ($k^{n_s-1}$)')
    plt.plot(k_physical, np.log(1e10*P_observe_smooth_highK), 'r.-', linewidth=2, label='Double-BC Observed PPS')
    plt.xscale('log')

    plt.xlabel(r'Wavenumber $k$ [Mpc$^{-1}$]', fontsize=13)
    plt.ylabel(r'Power Spectrum $P(k)$', fontsize=13)
    plt.title('Prediction: Geometric Modulation of the Power Spectrum', fontsize=14)
    plt.legend(fontsize=13)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f'{figures_dir}/PPS_power_spectra_cca_2D_dk_penalty_smooth_highK_As_ns_tau_bestfit.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot 1 to {figures_dir}/PPS_power_spectra_cca_2D_dk_penalty_smooth_highK.pdf")
    plt.show()

    # --- Plot 2: The Ratio (The "Signature") ---
    # This is the plot to put in your paper. It shows the deviation.
    ratio = P_observe_smooth_highK / P_standard

    plt.figure(figsize=(8, 4.5))
    plt.plot(k_physical, ratio, 'b.-', linewidth=2)
    plt.xscale('log')
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)

    plt.xlabel(r'Wavenumber $k$ [Mpc$^{-1}$]', fontsize=13)
    plt.ylabel(r'Ratio $P_{obs} / P_{std}$', fontsize=13)
    plt.title('Ratio of Double-BC PPS to Standard Power Law', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add text annotation explaining the low-k behavior
    if len(k_vals_int) > 5:
        plt.annotate('Geometric Suppression?',
                     xy=(k_physical[0], ratio[0]),
                     xytext=(k_physical[2], ratio[0]*0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig(f'{figures_dir}/PPS_ratio_cca_2D_dk_penalty_smooth_highK_As_ns_tau_bestfit.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot 2 to {figures_dir}/PPS_ratio_cca_2D_dk_penalty_smooth_highK_As_ns_tau_bestfit.pdf")
    plt.show()

    # # --- Plot 3: The Mixing Matrix (Visualizing the Leakage) ---
    # plt.figure(figsize=(8, 6))
    # plt.imshow(np.abs(eigenvecs_sparse_1)**2, cmap='Viridis', aspect='auto', interpolation='nearest')
    # plt.colorbar(label=r'Power Contribution $|\langle k | \Psi_\lambda \rangle|^2$')
    # plt.xlabel(r'Physical Mode Index $\lambda$')
    # plt.ylabel(r'Observed Fourier Mode $k$')
    # plt.title('Mode Mixing Matrix')
    # plt.show()