import numpy as np
import os
import re
from scipy.optimize import root_scalar
from anesthetic import read_chains

# Configuration
data_folder = '/home/wnd22/rds/hpc-work/Will_CPT/Higher_Order_Finding_U_Matrices/find_integer_allowedK/cosmo_param_direc/data/'
top_n = 10  # Number of top folders to analyze

# Planck data folder
planck_chain_path = '../Planck_data/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE'

#################
# Parameters
#################
Omega_gamma_h2 = 2.47e-5  # photon density
Neff = 3.046

################
# Main Analysis
################

def find_top_n_folders(n=10):
    """
    Find the top N folders with minimum slope_loss
    Returns: list of tuples (slope_loss, num, params)
    """
    data = []

    # Assuming 'data_folder' is defined in your broader scope as in your original code
    if os.path.exists(data_folder):
        master_log_file = f'{data_folder}/master_log.txt'

        if os.path.exists(master_log_file):
            try:
                with open(master_log_file, 'r') as f:
                    for num, line in enumerate(f):
                        parts = line.strip().split()
                        
                        # Make sure the line has at least the expected number of columns
                        if len(parts) >= 8:
                            # According to the format:
                            # [0] total_loss, [1] slope_loss, [2] per_k_integer_loss, [3] local_chi2
                            # [4] OmegaM, [5] OmegaK, [6] omega_b_ratio, [7] h, [8] k_array_str
                            slope_loss = float(parts[1])
                            
                            OmegaM = float(parts[4])
                            OmegaK = float(parts[5])
                            omega_b_ratio = float(parts[6])
                            h = float(parts[7])
                            
                            params = [OmegaM, OmegaK, omega_b_ratio, h]
                            data.append((slope_loss, num, params))
                            
            except Exception as e:
                print(f"Warning: Could not read {master_log_file}: {e}")

    # Sort by slope_loss (which is index 0 in our tuple) and get top N
    data.sort(key=lambda x: x[0])
    return data[:n]


def find_closest_mcmc_point(ns, target_omegam, target_omegabh2, target_h, target_omegak,
                             tolerance_omegam=0.005, tolerance_omegabh2=0.0005,
                             tolerance_h=0.005, tolerance_omegak=0.001):
    """
    Find MCMC points close to target values and return the one with minimum chi2_CMB

    Parameters:
    -----------
    ns : anesthetic samples
        Planck MCMC chains
    target_omegam : float
        Target Omega_m
    target_omegabh2 : float
        Target omega_b * h^2
    target_h : float
        Target h (Hubble parameter / 100)
    target_omegak : float
        Target Omega_K
    tolerance_* : float
        Tolerance for each parameter

    Returns:
    --------
    best_point : dict or None
        Dictionary with parameters and chi2_CMB, or None if no points found
    filtered_count : int
        Number of filtered points
    """

    # Filter points close to target values
    mask = (
        (np.abs(ns['omegam'] - target_omegam) <= tolerance_omegam) &
        (np.abs(ns['omegabh2'] - target_omegabh2) <= tolerance_omegabh2) &
        (np.abs(ns['H0']/100 - target_h) <= tolerance_h) &
        (np.abs(ns['omegak'] - target_omegak) <= tolerance_omegak)
    )

    filtered_ns = ns[mask]
    filtered_count = len(filtered_ns)

    if filtered_count == 0:
        return None, 0

    # Find the point with minimum chi2_CMB
    # Note: In Planck chains, the likelihood is stored as -log(L) = chi2/2
    # So we want the minimum value of chi2
    min_chi2_idx = filtered_ns['chi2_CMB'].idxmin()
    best_row = filtered_ns.loc[min_chi2_idx]

    best_point = {
        'chi2_CMB': best_row['chi2_CMB'],
        'omegam': best_row['omegam'],
        'omegabh2': best_row['omegabh2'],
        'omegak': best_row['omegak'],
        'H0': best_row['H0'],
        'h': best_row['H0']/100,
        'A_s': best_row['A'],
        'n_s': best_row['ns'],
        'tau': best_row['tau'],
        'omegal': best_row['omegal']
    }

    return best_point, filtered_count


def main():
    print(f"Searching for top {top_n} folders in {data_folder}...")
    print("="*80)

    # Find top N folders
    top_data_points = find_top_n_folders(top_n)

    if not top_data_points:
        print("No folders found!")
        return

    print(f"\nFound {len(top_data_points)} to analyze\n")

    # Load Planck MCMC chains
    print("Loading Planck MCMC chains...")
    ns = read_chains(planck_chain_path)
    print(f"Loaded {len(ns)} MCMC samples\n")

    # Results storage
    results = []

    # For each top folder, find closest MCMC point
    for i, (loss_integer, num, params) in enumerate(top_data_points):
        print(f"\n{'='*80}")
        print(f"Analyzing folder {i+1}/{len(top_data_points)}: try_{num}")
        print(f"{'='*80}")

        Omega_m,Omega_K, omega_b_ratio, h = params
        print(f"Loss integer: {loss_integer:.6f}")
        print(f"Parameters from file: mt={Omega_m:.4f}, kt={Omega_K:.4f}, omega_b_ratio={omega_b_ratio:.6f}, h={h:.4f}")

        # Calculate cosmological parameters
        omegabh2 = omega_b_ratio * Omega_m * h**2
        OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
        Omega_lambda = 1 - Omega_m - Omega_K - OmegaR

        print(f"\nCalculated cosmological parameters:")
        print(f"  Omega_m = {Omega_m:.6f}")
        print(f"  omegabh2 = {omegabh2:.6f}")
        print(f"  h = {h:.6f}")
        print(f"  Omega_K = {Omega_K:.6f}")
        print(f"  Omega_lambda = {Omega_lambda:.6f}")

        # Find closest MCMC point
        print("\nSearching for close MCMC points...")
        best_point, filtered_count = find_closest_mcmc_point(
            ns, Omega_m, omegabh2, h, Omega_K
        )

        if best_point is not None:
            print(f"Found {filtered_count} MCMC points within tolerance")
            print(f"\nBest-fit MCMC point (minimum chi2_CMB):")
            print(f"  chi2_CMB = {best_point['chi2_CMB']:.4f}")
            print(f"  A_s = {best_point['A_s']:.6f}")
            print(f"  n_s = {best_point['n_s']:.6f}")
            print(f"  tau = {best_point['tau']:.6f}")
            print(f"  H0 = {best_point['H0']:.4f}")
            print(f"  omegam = {best_point['omegam']:.6f}")
            print(f"  omegabh2 = {best_point['omegabh2']:.6f}")
            print(f"  omegak = {best_point['omegak']:.6f}")
            print(f"  omegak = {best_point['omegal']:.6f}")
            results.append({
                'folder_num': num,
                'loss_integer': loss_integer,
                'Omega_m_bestfit': Omega_m,
                'Omega_K_bestfit': Omega_K,
                'omega_b_ratio_bestfit': omega_b_ratio,
                'h_bestfit': h,
                'chi2_CMB': best_point['chi2_CMB'],
                'A_s': best_point['A_s'],
                'n_s': best_point['n_s'],
                'tau': best_point['tau'],
                'H0': best_point['H0'],
                'Omega_m': best_point['omegam'],
                'omegabh2': best_point['omegabh2'],
                'Omega_K': best_point['omegak'],
                'Omega_lambda': best_point['omegal'],
                'filtered_count': filtered_count
            })
        else:
            print("No MCMC points found within tolerance. Try increasing tolerance.")

    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Folder':<10} {'Loss_int':<12} {'chi2_CMB':<12} {'N_points':<10} {'Omega_K':<12}")
    print("-"*80)

    for i, result in enumerate(results):
        print(f"{i+1:<6} try_{result['folder_num']:<7} {result['loss_integer']:<12.6f} "
              f"{result['chi2_CMB']:<12.4f} {result['filtered_count']:<10} {result['Omega_K']:<12.6f}")

    # Save detailed results to file
    output_file = 'top10_mcmc_bestfit_results_param_direc.txt'
    with open(output_file, 'w') as f:
        f.write(f"Top {top_n} folders with MCMC best-fit parameters\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(results):
            f.write(f"\nRank {i+1}: try_{result['folder_num']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Loss integer: {result['loss_integer']:.6f}\n")
            f.write(f"Number of MCMC points within tolerance: {result['filtered_count']}\n")
            f.write(f"chi2_CMB: {result['chi2_CMB']:.4f}\n")
            f.write(f"\nModel parameters (from loss_params file):\n")
            f.write(f"  Omega_m = {result['Omega_m_bestfit']:.4f}\n")
            f.write(f"  Omega_K = {result['Omega_K_bestfit']:.4f}\n")
            f.write(f"  omega_b_ratio = {result['omega_b_ratio_bestfit']:.6f}\n")
            f.write(f"  h = {result['h_bestfit']:.6f}\n")
            f.write(f"\nBest-fit MCMC parameters:\n")
            f.write(f"  A_s = {result['A_s']:.6f}\n")
            f.write(f"  n_s = {result['n_s']:.6f}\n")
            f.write(f"  tau = {result['tau']:.6f}\n")
            f.write(f"  H0 = {result['H0']:.4f}\n")
            f.write(f"\nCosmological parameters:\n")
            f.write(f"  Omega_m = {result['Omega_m']:.6f}\n")
            f.write(f"  omegabh2 = {result['omegabh2']:.6f}\n")
            f.write(f"  Omega_K = {result['Omega_K']:.6f}\n")
            f.write(f"  Omega_lambda = {result['Omega_lambda']:.6f}\n")

    print(f"\nResults saved to {output_file}")

    # Also save to CSV for easy analysis
    csv_file = 'top10_mcmc_bestfit_results_param_direc.csv'
    with open(csv_file, 'w') as f:
        # Header
        f.write("rank,folder_num,loss_integer,chi2_CMB,filtered_count,Omega_m_bestfit,Omega_K_bestfit,omega_b_ratio_bestfit,h_bestfit,")
        f.write("A_s,n_s,tau,H0,Omega_m,omegabh2,Omega_K,Omega_lambda\n")

        # Data
        for i, result in enumerate(results):
            f.write(f"{i+1},{result['folder_num']},{result['loss_integer']:.6e},{result['chi2_CMB']:.4f},")
            f.write(f"{result['filtered_count']},{result['Omega_m_bestfit']:.6f},{result['Omega_K_bestfit']:.6f},")
            f.write(f"{result['omega_b_ratio_bestfit']:.6f},{result['h_bestfit']:.6f},{result['A_s']:.6f},")
            f.write(f"{result['n_s']:.6f},{result['tau']:.6f},{result['H0']:.4f},")
            f.write(f"{result['Omega_m']:.6f},{result['omegabh2']:.6f},{result['Omega_K']:.6f},")
            f.write(f"{result['Omega_lambda']:.6f}\n")

    print(f"CSV results saved to {csv_file}")

if __name__ == "__main__":
    main()
