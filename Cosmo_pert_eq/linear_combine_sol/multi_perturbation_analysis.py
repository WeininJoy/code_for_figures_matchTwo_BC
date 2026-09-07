# -*- coding: utf-8 -*-
"""
This script performs a Gram-Schmidt orthonormalization and eigenvalue analysis
on two different bases of the COMPLETE cosmological perturbation solutions for
the radiation velocity, v_r(eta).

The solution generator is based on the user's verified script for plotting
the full palindromic evolution.

- basis_1 is constructed from wavenumbers 'k' in a closed universe model.
- basis_2 is constructed from the 'allowed' wavenumbers for a palindromic flat universe.

It identifies and compares the common eigenfunctions of v_r found in both bases.
"""
import json
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# =============================================================================
# 1. SETUP: Parameters and Constants (Mostly Unchanged)
# =============================================================================
nu_spacing = 4

print("--- Setting up parameters and functions ---")

## --- Best-fit parameters ---
lam = 1
rt = 1
Omega_gamma_h2 = 2.47e-5 # photon density 
Neff = 3.046

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
    return Omega_lambda, Omega_m, Omega_K

###############################################################################
# params with integerK and best-fit with observation
mt, kt, Omegab_ratio, h, As, ns, tau = 426.32052836334015, 1.5217932160492045, 0.1555991124300318, 0.5567037724095327, 2.030476,0.967708,0.046112 # k50-65_refined_1

OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
z_rec = 1063.4075 # calculated based on the calculate_z_rec() output
###############################################################################

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#set tolerances
atol = 1e-13;
rtol = 1e-13;
stol = 1e-10;
num_variables = 75; # number of pert variables
swaptime = 2; #set time when we swap from s to sigma
deltaeta = 6.6e-4
H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
Hinf = H0*np.sqrt(OmegaLambda);
a0=1; K=-OmegaK * a0**2 * H0**2

#```````````````````````````````````````````````````````````````````````````````
#BACKGROUND EQUATIONS
#```````````````````````````````````````````````````````````````````````````````

#write derivative function for background
def ds_dt(t, s):
    return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

def da_dt(t, a):
    return a**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(((a**2))) + OmegaM/abs(((a**3))) + OmegaR/abs((a**4))))

t0 = 1e-5;

#set coefficients for initial conditions
# smin1 = np.sqrt(3*OmegaLambda/OmegaR);
# szero = - OmegaM/(4*OmegaR);
# s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR));
# s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR) ;
# s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)));
# s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

# s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4;

a1 = np.sqrt(OmegaR)/(np.sqrt(3)*np.sqrt(OmegaLambda));
a2 = OmegaM/(12*OmegaLambda);
a3 = (OmegaK * np.sqrt(OmegaR))/(18 * np.sqrt(3) * OmegaLambda**(3/2));
a4 = (OmegaK * OmegaM)/(432 * OmegaLambda**2);
a5 = (np.sqrt(OmegaR) * (OmegaK**2 + 12 * OmegaR * OmegaLambda))/(1080 * np.sqrt(3) * OmegaLambda**(5/2));
a6 = (OmegaM * (OmegaK**2 + 72 * OmegaR * OmegaLambda))/(38880 * OmegaLambda**3);
a_Bang = a1*t0 + a2*t0**2 + a3*t0**3 + a4*t0**4 + a5*t0**5 + a6*t0**6; 

print('Performing Initial Background Integration')
def reach_FCB(t, s): return s[0]
reach_FCB.terminal = True

sol_a = solve_ivp(da_dt, [t0,swaptime], [a_Bang], max_step = 0.25e-4, method='LSODA', atol=atol, rtol=rtol)
sol = solve_ivp(ds_dt, [swaptime, 12], [1./sol_a.y[0][-1]], max_step = 0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
print('Initial Background Integration Done')

# Check if t_events[0] is not empty before trying to access its elements
if sol.t_events and len(sol.t_events[0]) > 0:
    fcb_time = sol.t_events[0][0]
    print(f"fcb_time: {fcb_time}")
else:
    print(f"Event 'reach_FCB' did not occur.")
    # You might want to assign a default value or 'None' to fcb_time here
    fcb_time = None # Or np.nan, or some other indicator

# Rest of your code that uses fcb_time would go here
# For example:
if fcb_time is not None:
    print(f"Further processing with fcb_time = {fcb_time}")
else:
    print(f"No fcb_time available for further processing.")

endtime = fcb_time - deltaeta

#``````````````````````````````````````````````````````````````````````````````
#RECOMBINATION CONFORMAL TIME
#```````````````````````````````````````````````````````````````````````````````

#find conformal time at recombination
a_rec = 1./(1+z_rec)  #reciprocal scale factor at recombination

#take difference between s values and s_rec to find where s=s_rec i.e where recScaleFactorDifference=0
recScaleFactorDifference = abs(sol_a.y[0] - a_rec) #take difference between s values and s_rec to find where s=s_rec
recConformalTime = sol_a.t[recScaleFactorDifference.argmin()]
print(f"Recombination conformal time: {recConformalTime}")

# Perfect fluid ODE for early times (unchanged)
def dX_perfect_sigma(t, X, k):
    sigma, phi, dr, dm, vr, vm = X
    sigmadot = -(H0)*np.sqrt(OmegaLambda*np.exp(-2*sigma)+OmegaM*np.exp(sigma)+OmegaR*np.exp(2*sigma))
    rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
    rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))
    phidot = sigmadot*phi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
    drdot = (4/3)*(3*phidot + k**2*vr)
    dmdot = 3*phidot + k**2*vm
    vrdot = -(phi + dr/4) 
    vmdot = sigmadot*vm - phi
    return [sigmadot, phidot, drdot, dmdot, vrdot, vmdot]

def generate_multi_perturbation_bases(discreteK_type, eta_grid, folder_path='./data/',
                                       folder_path_matrices=None,
                                       folder_path_timeseries=None):
    # discreteK_type = 'allowedK' or 'integerK'
    # folder_path_matrices   : explicit path to the static-matrix folder.
    #                          If None, defaults to folder_path + 'data_{discreteK_type}/'.
    # folder_path_timeseries : explicit path to the timeseries folder.
    #                          If None, defaults to folder_path + 'data_{discreteK_type}_timeseries/'.
    #
    # Passing explicit paths lets callers point directly at a per-range chunk
    # folder (e.g. 'data_allowedK_chunks/k30_50/') without needing to merge
    # first, or at the merged 'data_allowedK/' folder for the full dataset.
    # =============================================================================
    # 2. DATA LOADING
    # =============================================================================
    # *** Point to the time-series data directory ***
    if folder_path_matrices is None:
        folder_path_matrices = folder_path + f'data_{discreteK_type}/'
    if folder_path_timeseries is None:
        folder_path_timeseries = folder_path + f'data_{discreteK_type}_timeseries/'

    print("\n--- Loading pre-computed time-dependent transfer matrices ---")
    try:
        # *** NEW: Load the time-series data ***
        t_grid = np.load(folder_path_timeseries + 't_grid.npy')
        allowedK = np.load(folder_path_timeseries + 'L70_kvalues.npy')
        all_ABC_solutions = np.load(folder_path_timeseries + 'L70_ABC_solutions.npy')
        all_DEF_solutions = np.load(folder_path_timeseries + 'L70_DEF_solutions.npy')
        all_GHI_solutions = np.load(folder_path_timeseries + 'L70_GHI_solutions.npy')
        
        # Load the static matrices needed for boundary conditions
        ABCmatrices = np.load(folder_path_matrices+'L70_ABCmatrices.npy')
        DEFmatrices = np.load(folder_path_matrices+'L70_DEFmatrices.npy')
        GHIvectors = np.load(folder_path_matrices+'L70_GHIvectors.npy')
        X1matrices = np.load(folder_path_matrices + 'L70_X1matrices.npy')
        X2matrices = np.load(folder_path_matrices + 'L70_X2matrices.npy')
        recValues = np.load(folder_path_matrices + 'L70_recValues.npy')
        
        print(f"Loaded solution histories for {len(allowedK)} allowed K values.")
        print(f"Time grid has {len(t_grid)} points from eta={t_grid[0]:.2f} to eta={t_grid[-1]:.2f}")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have run 'Higher_Order_Finding_U_Matrices_TimeSeries.py' first.")
        exit()

    # Use the actual loaded num_variables (may differ across k-range chunks after padding)
    loaded_nv = ABCmatrices.shape[1]

    # Extract all matrix components
    Amatrices = ABCmatrices[:, 0:6, :]
    Bmatrices = ABCmatrices[:, 6:8, :]
    Cmatrices = ABCmatrices[:, 8:loaded_nv, :]
    Dmatrices = DEFmatrices[:, 0:6, :]
    Ematrices = DEFmatrices[:, 6:8, :]
    Fmatrices = DEFmatrices[:, 8:loaded_nv, :]

    print(f"Loaded num_variables={loaded_nv} (padded to largest chunk size)")
    print(f"Matrix shapes: A={Amatrices.shape}, B={Bmatrices.shape}, C={Cmatrices.shape}")
    print(f"               D={Dmatrices.shape}, E={Ematrices.shape}, F={Fmatrices.shape}")

    # =============================================================================
    # 3. CALCULATE AND RECONSTRUCT SOLUTIONS
    # =============================================================================

    print("\n--- Reconstructing solutions for the first 3 allowed modes ---")

    perturbation_types = ['dr', 'dm', 'vr', 'vm']  # Note: phi and psi are not directly considered, since they can be derived from other variables
    basis_dict = {pert_type: [] for pert_type in perturbation_types}

    for i in range(len(allowedK)):
        # This logic now selects the first few modes for simplicity
        k_index = i 
        k = allowedK[k_index]
        print(f"\nProcessing mode n={i+1} with k={k:.6f} (index {k_index})")

        # --- a) Get matrices for solving boundary conditions ---
        # *** NEW: Extract endpoint matrices from the time-series data ***
        # The saved arrays are from rec_time to end_time, so the rec_time values are at index 0.

        # Get exact matrices for this k
        A = Amatrices[k_index]
        B = Bmatrices[k_index] 
        C = Cmatrices[k_index]
        D = Dmatrices[k_index]
        E = Ematrices[k_index]
        F = Fmatrices[k_index]
        X1 = X1matrices[k_index]
        X2 = X2matrices[k_index]
        recs_vec = recValues[k_index]

        print(f"Matrix shapes for k={k:.6f}:")
        print(f"A: {A.shape}, B: {B.shape}, C: {C.shape}")
        print(f"D: {D.shape}, E: {E.shape}, F: {F.shape}")
        print(f"X1: {X1.shape}, X2: {X2.shape}")
        print(f"Stored recValues: {recs_vec}")

        # Calculate x^∞ 
        GX3 = np.zeros((6,4))
        GX3[:,2] = GHIvectors[k_index][0:6]

        M_matrix = (A @ X1 + D @ X2)[2:6, :]  # Only use independent components [dr, dm, vr, vm]
        x_rec = recs_vec[2:6]  # Only use independent components from stored values
        # x_inf = np.linalg.solve(M_matrix, x_rec)
        x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]
        print(f"x^∞ = {x_inf}")
        # x_inf[2] = 0

        # Calculate x' and y' at endtime using X1 and X2 (equation 25)
        x_prime_coeffs = X1 @ x_inf
        y_prime_coeffs = X2 @ x_inf

        # --- d) RECONSTRUCT solution from recTime to endTime ---
        # *** THIS ENTIRE SECTION REPLACES THE BACKWARD INTEGRATION ***
        print("  Reconstructing solution from pre-computed basis...")
        ABC_sols_k = all_ABC_solutions[k_index] # Shape: (num_vars, 6, num_times)
        DEF_sols_k = all_DEF_solutions[k_index] # Shape: (num_vars, 2, num_times)
        GHI_sols_k = all_GHI_solutions[k_index] # Shape: (num_vars, num_times)
        
        # Linearly combine the basis solutions using the calculated coefficients
        # Y(t) = U_ABC(t) * x_prime_coeffs + U_DEF(t) * y_prime_coeffs + Y_GHI(t)
        Y_reconstructed = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                        np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs) # + GHI_sols_k

        # The time array for this solution is simply the loaded t_grid
        t_backward = t_grid
        
        # Add the background 's' variable back for stitching and plotting
        s_background = np.interp(t_grid, sol.t, sol.y[0])
        Y_backward = np.vstack([s_background, Y_reconstructed])
        
        # # --- e) Get solution from Big Bang to Recombination (unchanged) ---
        # # (Using a simplified initial condition for this example)
        # phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
        # phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
        
        # dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
        # dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda);
        
        # dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
        # dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda);
        
        # vr1 = -1/2;
        # vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
        # vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda);
        
        # vm1 = -1/2;
        # vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
        # vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda);
        
        # #set initial conditions
        # t0 = 1e-8;
        # s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3;
        # sigma0 = np.log(s0)
        # phi0 = 1 + phi1*t0 + phi2*t0**2; #t0 from above in "background equations section"
        # dr0 = -2 + dr1*t0 + dr2*t0**2;
        # dm0 = -1.5 + dm1*t0 + dm2*t0**2;
        # vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3;
        # vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3;

        # Y0_perfect = [sigma0, phi0, dr0, dm0, vr0, vm0]
        # sol_perfect = solve_ivp(dX_perfect_sigma, [t0, recConformalTime], Y0_perfect,
        #                         dense_output=True, method='LSODA', atol=atol, rtol=rtol, args=(k,))

        # # --- f) Stitch and Reflect ---
        # # *** MODIFIED SECTION: PREPARE AND STORE HALVES SEPARATELY ***

        # Y_perfect_full = np.zeros((num_variables + 1, len(sol_perfect.t)))
        # Y_perfect_full[0,:] = np.exp(sol_perfect.y[0,:])
        # Y_perfect_full[1,:] = sol_perfect.y[1,:]
        # Y_perfect_full[2,:] = sol_perfect.y[1,:]
        # Y_perfect_full[3:7,:] = sol_perfect.y[2:,:]

        # First, create the common time array
        # t_full_unsorted = np.concatenate((sol_perfect.t, t_backward[::-1], [fcb_time]))
        t_full_unsorted = np.concatenate((t_backward[::-1], [fcb_time]))

        # Now, assemble each solution vector individually
        # Note: Indices come from your original if/elif block
        # Perfect fluid state: [sigma, phi, dr, dm, vr, vm] -> indices 0, 1, 2, 3, 4, 5
        # Boltzmann state: [s, phi, phidot, dr, dm, vr, vm, ...] -> indices 0, 1, 2, 3, 4, 5, 6
        
        solutions_unsorted = {
            'dr': np.concatenate((
                # sol_perfect.y[2, :],          # Perfect fluid dr
                Y_backward[3, ::-1],          # Boltzmann dr
                [x_inf[0]]                    # Value at FCB (dr_inf)
            )),
            'dm': np.concatenate((
                # sol_perfect.y[3, :],          # Perfect fluid dm
                Y_backward[4, ::-1],          # Boltzmann dm
                [x_inf[1]]                    # Value at FCB (dm_inf)
            )),
            'vr': np.concatenate((
                # sol_perfect.y[4, :],          # Perfect fluid vr
                Y_backward[5, ::-1],          # Boltzmann vr
                [x_inf[2]]                    # Value at FCB (vr_inf)
            )),
            'vm': np.concatenate((
                # sol_perfect.y[5, :],          # Perfect fluid vm
                Y_backward[6, ::-1],          # Boltzmann vm
                [(X1 @ x_inf)[3]]             # Value at FCB (vm_inf)
            ))
        }
        
        # Sort by time to ensure monotonicity for interpolation
        sort_indices = np.argsort(t_full_unsorted)
        t_sol = t_full_unsorted[sort_indices]
        
        # Apply the same sorting to all solution arrays in the dictionary
        y_sol = {key: value[sort_indices] for key, value in solutions_unsorted.items()}
            
        # Interpolate each perturbation type onto common eta grid
        for pert_type in perturbation_types:
            if pert_type in y_sol:
                interpolator = interp1d(t_sol, y_sol[pert_type], 
                                        bounds_error=False, fill_value=0.0)
                solution = interpolator(eta_grid)
                
                basis_dict[pert_type].append(solution)
            else:
                print(f"    Warning: {pert_type} not found in solution, using zeros")
                basis_dict[pert_type].append(np.zeros_like(eta_grid))
    
    # Convert lists to numpy arrays
    for pert_type in perturbation_types:
        basis_dict[pert_type] = np.array(basis_dict[pert_type]).T  # Transpose for QR

    return basis_dict

# =============================================================================
# SECTION 2: GRAM-SCHMIDT ANALYSIS 
# =============================================================================
def qr_decomposition(basis):
    """QR decomposition with proper handling"""
    orthonormal_functions, transformation_matrix = np.linalg.qr(basis)
    return orthonormal_functions, np.linalg.inv(transformation_matrix.T)

def compute_multi_perturbation_A_matrix(ortho_funcs_1_dict, ortho_funcs_2_dict):
    """
    Compute the A matrix by summing contributions from all perturbation types
    
    Parameters:
    ortho_funcs_1_dict: dict of orthonormal functions from basis 1 for each perturbation
    ortho_funcs_2_dict: dict of orthonormal functions from basis 2 for each perturbation
    
    Returns:
    Combined eigenvalues and eigenvectors from all perturbations
    """
    perturbation_types = ['dr', 'dm', 'vr', 'vm'] # only those the oscillating modes
    
    # Initialize combined M matrix
    M_total = None
    n_perturbations = 0
    
    print("Computing combined A matrix from all perturbations...")
    
    for pert_type in perturbation_types:
        if pert_type in ortho_funcs_1_dict and pert_type in ortho_funcs_2_dict:
            # Compute M for this perturbation type
            M_pert = np.dot(ortho_funcs_1_dict[pert_type].T, ortho_funcs_2_dict[pert_type])
            
            if M_total is None:
                M_total = M_pert
            else:
                M_total += M_pert
            
            n_perturbations += 1
            print(f"  Added contribution from {pert_type}")
    
    # Normalize by the number of perturbations
    if n_perturbations > 0:
        M_total = M_total / n_perturbations
        print(f"  Normalized by {n_perturbations} perturbation types")
    
    # Compute eigenvalues and eigenvectors of combined matrix
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(np.dot(M_total, M_total.T))
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(np.dot(M_total.T, M_total))
    
    return eigenvalues_1, eigenvectors_1, eigenvalues_2, eigenvectors_2, M_total

def choose_eigenvalues(eigenvalues, eigenvectors, eigenvalues_threshold=0.99):
    """Select the largest eigenvalues"""
    # Sort eigenvalues in descending order and get indices
    sorted_indices = np.argsort(np.real(eigenvalues))[::-1]
    
    eigenvalues_valid, eigenvectors_valid = [], []
    for i in range(len(eigenvalues)):
        idx = sorted_indices[i]
        if np.real(eigenvalues[idx]) < eigenvalues_threshold:
            break
        eigenvalues_valid.append(eigenvalues[idx])
        eigenvectors_valid.append(eigenvectors[:, idx])
    
    return eigenvalues_valid, eigenvectors_valid

def compute_coefficients(eigenvalues, eigenvectors, transformation_matrix):
    """Compute linear combination coefficients"""
    coefficients = np.zeros((len(eigenvalues), transformation_matrix.shape[1]), dtype=float)
    for i in range(len(eigenvalues)):
        coefficients[i, :] = np.dot(np.array(eigenvectors[i]), transformation_matrix)
    return coefficients

def _pad_to_max_nv(arrays, axis=1):
    """Zero-pad arrays along `axis` so all have the same size as the largest."""
    max_size = max(a.shape[axis] for a in arrays)
    padded = []
    for a in arrays:
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (0, max_size - a.shape[axis])
        padded.append(np.pad(a, pad_width))
    return padded


def load_and_merge_chunked_highk_data(folder_path, eta_grid):
    """
    Load and merge high-K data from chunked files.

    Parameters:
    -----------
    folder_path : str
        Base path to data folders
    eta_grid : numpy array
        Time grid for interpolation

    Returns:
    --------
    basis_highk_dict : dict
        Dictionary containing merged high-K perturbation bases for each type
        Returns None if no chunk data is found
    """
    highk_base_path = folder_path + 'data_highK_integerK/'
    highk_timeseries_base_path = folder_path + 'data_highK_integerK_timeseries/'

    if not os.path.exists(highk_base_path):
        print(f"\nHigh-K base directory not found: {highk_base_path}")
        return None

    print("\nLoading high-K chunked data...")

    # Find all chunk directories
    chunk_dirs = sorted([d for d in os.listdir(highk_base_path)
                       if os.path.isdir(os.path.join(highk_base_path, d)) and d.startswith('chunk_')])

    if len(chunk_dirs) == 0:
        print(f"  No chunk directories found in {highk_base_path}")
        return None

    print(f"Found {len(chunk_dirs)} chunk directories: {chunk_dirs}")

    # Initialize lists to accumulate data from all chunks
    all_chunks_kvalues = []
    all_chunks_ABC_solutions = []
    all_chunks_DEF_solutions = []
    all_chunks_GHI_solutions = []
    all_chunks_ABCmatrices = []
    all_chunks_DEFmatrices = []
    all_chunks_GHIvectors = []
    all_chunks_X1matrices = []
    all_chunks_X2matrices = []
    all_chunks_recValues = []
    t_grid_highk = None

    # Load and merge data from each chunk
    for chunk_dir in chunk_dirs:
        chunk_path = os.path.join(highk_base_path, chunk_dir) + '/'
        chunk_timeseries_path = os.path.join(highk_timeseries_base_path, chunk_dir) + '/'

        try:
            print(f"  Loading chunk: {chunk_dir}")

            # Load timeseries data
            chunk_t_grid = np.load(chunk_timeseries_path + 't_grid.npy')
            chunk_kvalues = np.load(chunk_timeseries_path + 'L70_kvalues.npy')
            chunk_ABC_solutions = np.load(chunk_timeseries_path + 'L70_ABC_solutions.npy')
            chunk_DEF_solutions = np.load(chunk_timeseries_path + 'L70_DEF_solutions.npy')
            chunk_GHI_solutions = np.load(chunk_timeseries_path + 'L70_GHI_solutions.npy')

            # Load static matrices
            chunk_ABCmatrices = np.load(chunk_path + 'L70_ABCmatrices.npy')
            chunk_DEFmatrices = np.load(chunk_path + 'L70_DEFmatrices.npy')
            chunk_GHIvectors = np.load(chunk_path + 'L70_GHIvectors.npy')
            chunk_X1matrices = np.load(chunk_path + 'L70_X1matrices.npy')
            chunk_X2matrices = np.load(chunk_path + 'L70_X2matrices.npy')
            chunk_recValues = np.load(chunk_path + 'L70_recValues.npy')

            # Verify t_grid consistency across chunks
            if t_grid_highk is None:
                t_grid_highk = chunk_t_grid
            else:
                if not np.allclose(t_grid_highk, chunk_t_grid):
                    print(f"    Warning: t_grid mismatch in {chunk_dir}, using first chunk's t_grid")

            # Append to lists
            all_chunks_kvalues.append(chunk_kvalues)
            all_chunks_ABC_solutions.append(chunk_ABC_solutions)
            all_chunks_DEF_solutions.append(chunk_DEF_solutions)
            all_chunks_GHI_solutions.append(chunk_GHI_solutions)
            all_chunks_ABCmatrices.append(chunk_ABCmatrices)
            all_chunks_DEFmatrices.append(chunk_DEFmatrices)
            all_chunks_GHIvectors.append(chunk_GHIvectors)
            all_chunks_X1matrices.append(chunk_X1matrices)
            all_chunks_X2matrices.append(chunk_X2matrices)
            all_chunks_recValues.append(chunk_recValues)

            print(f"    Loaded {len(chunk_kvalues)} k-modes from {chunk_dir}")

        except FileNotFoundError as e:
            print(f"    Warning: Could not load chunk {chunk_dir}: {e}")
            continue

    # Concatenate all chunks along the k-mode axis (axis=0)
    if len(all_chunks_kvalues) == 0:
        print(f"  Warning: No chunks were successfully loaded")
        return None

    print(f"\n  Merging {len(all_chunks_kvalues)} chunks...")

    # Chunks may have different num_variables (larger for high-k ranges).
    # Zero-pad along axis 1 (num_variables axis) before concatenating so that
    # the merged arrays have a uniform shape equal to the largest chunk's size.
    merged_kvalues = np.concatenate(all_chunks_kvalues, axis=0)
    merged_ABC_solutions = np.concatenate(_pad_to_max_nv(all_chunks_ABC_solutions, axis=1), axis=0)
    merged_DEF_solutions = np.concatenate(_pad_to_max_nv(all_chunks_DEF_solutions, axis=1), axis=0)
    merged_GHI_solutions = np.concatenate(_pad_to_max_nv(all_chunks_GHI_solutions, axis=1), axis=0)
    merged_ABCmatrices = np.concatenate(_pad_to_max_nv(all_chunks_ABCmatrices, axis=1), axis=0)
    merged_DEFmatrices = np.concatenate(_pad_to_max_nv(all_chunks_DEFmatrices, axis=1), axis=0)
    merged_GHIvectors = np.concatenate(_pad_to_max_nv(all_chunks_GHIvectors, axis=1), axis=0)
    merged_X1matrices = np.concatenate(all_chunks_X1matrices, axis=0)
    merged_X2matrices = np.concatenate(all_chunks_X2matrices, axis=0)
    merged_recValues = np.concatenate(all_chunks_recValues, axis=0)

    print(f"  Total merged k-modes: {len(merged_kvalues)}")
    print(f"  K range: {merged_kvalues[0]:.6f} to {merged_kvalues[-1]:.6f}")

    # Now reconstruct perturbation bases from merged chunk data
    print(f"  Reconstructing perturbation solutions from merged chunks...")

    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    basis_highk_dict = {pert_type: [] for pert_type in perturbation_types}

    # Use the actual num_variables from the merged (padded) arrays
    merged_nv = merged_ABCmatrices.shape[1]

    # Extract matrix components
    Amatrices_highk = merged_ABCmatrices[:, 0:6, :]
    Bmatrices_highk = merged_ABCmatrices[:, 6:8, :]
    Cmatrices_highk = merged_ABCmatrices[:, 8:merged_nv, :]
    Dmatrices_highk = merged_DEFmatrices[:, 0:6, :]
    Ematrices_highk = merged_DEFmatrices[:, 6:8, :]
    Fmatrices_highk = merged_DEFmatrices[:, 8:merged_nv, :]

    for i in range(len(merged_kvalues)):
        k = merged_kvalues[i]

        # Get matrices for this k
        A = Amatrices_highk[i]
        B = Bmatrices_highk[i]
        C = Cmatrices_highk[i]
        D = Dmatrices_highk[i]
        E = Ematrices_highk[i]
        F = Fmatrices_highk[i]
        X1 = merged_X1matrices[i]
        X2 = merged_X2matrices[i]
        recs_vec = merged_recValues[i]

        # Calculate x^∞
        GX3 = np.zeros((6,4))
        GX3[:,2] = merged_GHIvectors[i][0:6]

        M_matrix = (A @ X1 + D @ X2)[2:6, :]
        x_rec = recs_vec[2:6]
        x_inf = np.linalg.lstsq(M_matrix, x_rec, rcond=None)[0]

        # Calculate x' and y' coefficients
        x_prime_coeffs = X1 @ x_inf
        y_prime_coeffs = X2 @ x_inf

        # Reconstruct solution from basis
        ABC_sols_k = merged_ABC_solutions[i]
        DEF_sols_k = merged_DEF_solutions[i]
        GHI_sols_k = merged_GHI_solutions[i]

        Y_reconstructed = np.einsum('ijt,j->it', ABC_sols_k, x_prime_coeffs) + \
                        np.einsum('ijt,j->it', DEF_sols_k, y_prime_coeffs)

        # Interpolate background s
        s_background = np.interp(t_grid_highk, sol.t, sol.y[0])
        Y_backward = np.vstack([s_background, Y_reconstructed])

        # Construct full time array
        t_full_unsorted = np.concatenate((t_grid_highk[::-1], [fcb_time]))

        # Assemble solutions
        solutions_unsorted = {
            'dr': np.concatenate((Y_backward[3, ::-1], [x_inf[0]])),
            'dm': np.concatenate((Y_backward[4, ::-1], [x_inf[1]])),
            'vr': np.concatenate((Y_backward[5, ::-1], [x_inf[2]])),
            'vm': np.concatenate((Y_backward[6, ::-1], [(X1 @ x_inf)[3]]))
        }

        # Sort by time
        sort_indices = np.argsort(t_full_unsorted)
        t_sol = t_full_unsorted[sort_indices]
        y_sol = {key: value[sort_indices] for key, value in solutions_unsorted.items()}

        # Interpolate onto eta_grid
        for pert_type in perturbation_types:
            if pert_type in y_sol:
                interpolator = interp1d(t_sol, y_sol[pert_type],
                                      bounds_error=False, fill_value=0.0)
                solution = interpolator(eta_grid)
                basis_highk_dict[pert_type].append(solution)
            else:
                basis_highk_dict[pert_type].append(np.zeros_like(eta_grid))

    # Convert lists to numpy arrays
    for pert_type in perturbation_types:
        basis_highk_dict[pert_type] = np.array(basis_highk_dict[pert_type]).T

    print(f"  High-K chunk processing completed successfully!")
    return basis_highk_dict

def multi_perturbation_analysis_with_full_extension(N=23, N_t=500, folder_path='./data/', eigenvalues_threshold=0.99):
    """
    Complete multi-perturbation eigenvalue analysis INCLUDING extended K modes.

    This is the "naive" approach that merges extended modes BEFORE eigenvalue analysis.
    Warning: High-k perfect eigenvalues (1.0) may dominate low-k physics eigenvalues (~0.9999).
    Use for comparison with divide-and-conquer approach.
    """
    print(f"Starting multi-perturbation analysis WITH FULL EXTENSION with N={N}, N_t={N_t}")

    # Define eta_grid from cutoff_time to fcb_time as requested
    eta_grid = np.linspace(0, fcb_time, N_t)
    print(f"Using eta_grid from cutoff to FCB: eta ∈ [0, {fcb_time:.4e}]")

    # Generate bases
    print("\nGenerating basis 1 (Closed Universe)...")
    basis_1_dict = generate_multi_perturbation_bases("integerK", eta_grid, folder_path=folder_path)

    print("\nGenerating basis 2 (Palindromic Universe)...")
    basis_2_dict = generate_multi_perturbation_bases("allowedK", eta_grid, folder_path=folder_path)

    # Merge extended basis BEFORE eigenvalue analysis
    extend_data_path = folder_path + 'data_extend_integerK/'
    if os.path.exists(extend_data_path):
        print("\nMerging extended basis BEFORE eigenvalue analysis...")
        try:
            basis_extend_dict = generate_multi_perturbation_bases("extend_integerK", eta_grid, folder_path=folder_path)

            perturbation_types = ['dr', 'dm', 'vr', 'vm']
            for pert_type in perturbation_types:
                if pert_type in basis_extend_dict:
                    # Concatenate along the modes axis (axis=1)
                    basis_1_dict[pert_type] = np.concatenate([basis_1_dict[pert_type], basis_extend_dict[pert_type]], axis=1)
                    basis_2_dict[pert_type] = np.concatenate([basis_2_dict[pert_type], basis_extend_dict[pert_type]], axis=1)
                    print(f"  Extended {pert_type}: basis_1 shape = {basis_1_dict[pert_type].shape}, basis_2 shape = {basis_2_dict[pert_type].shape}")
        except FileNotFoundError as e:
            print(f"Warning: Extended basis data files not found: {e}")
    else:
        print(f"\nExtended basis directory not found: {extend_data_path}")

    # Merge high-K chunked data BEFORE eigenvalue analysis
    basis_highk_dict = load_and_merge_chunked_highk_data(folder_path, eta_grid)
    if basis_highk_dict is not None:
        print("\nMerging high-K chunked basis to basis_1 and basis_2...")
        perturbation_types = ['dr', 'dm', 'vr', 'vm']
        for pert_type in perturbation_types:
            if pert_type in basis_highk_dict:
                # Concatenate along the modes axis (axis=1)
                basis_1_dict[pert_type] = np.concatenate([basis_1_dict[pert_type], basis_highk_dict[pert_type]], axis=1)
                basis_2_dict[pert_type] = np.concatenate([basis_2_dict[pert_type], basis_highk_dict[pert_type]], axis=1)
                print(f"  High-K {pert_type}: basis_1 shape = {basis_1_dict[pert_type].shape}, basis_2 shape = {basis_2_dict[pert_type].shape}")

    # Perform QR decomposition on FULL (core + extended) bases
    print("\n--- Running Eigen-Analysis on FULL Bases (Core + Extended) ---")

    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    ortho_funcs_1_dict = {}
    ortho_funcs_2_dict = {}
    transform_1_dict = {}
    transform_2_dict = {}

    for pert_type in perturbation_types:
        if pert_type not in basis_1_dict:
            continue

        print(f"  QR decomposition for {pert_type} (full basis)")

        if np.all(basis_1_dict[pert_type] == 0):
            print(f"    Warning: Basis 1 for {pert_type} is all zeros, skipping")
            continue
        if np.all(basis_2_dict[pert_type] == 0):
            print(f"    Warning: Basis 2 for {pert_type} is all zeros, skipping")
            continue

        try:
            ortho_funcs_1_dict[pert_type], transform_1_dict[pert_type] = qr_decomposition(basis_1_dict[pert_type])
            ortho_funcs_2_dict[pert_type], transform_2_dict[pert_type] = qr_decomposition(basis_2_dict[pert_type])
            print(f"    Successfully processed {pert_type}")
        except Exception as e:
            print(f"    Error in QR decomposition for {pert_type}: {e}")

    # Compute eigenvalues on FULL basis (including extended modes)
    print("\nComputing combined eigenvalue analysis (FULL basis)...")
    eigenvals_1, eigenvecs_1, eigenvals_2, eigenvecs_2, M_combined = compute_multi_perturbation_A_matrix(
        ortho_funcs_1_dict, ortho_funcs_2_dict)

    # Filter valid eigenvalues
    eigenvals_valid_1, eigenvecs_valid_1 = choose_eigenvalues(
        eigenvals_1, eigenvecs_1, eigenvalues_threshold)
    eigenvals_valid_2, eigenvecs_valid_2 = choose_eigenvalues(
        eigenvals_2, eigenvecs_2, eigenvalues_threshold)

    print(f"\nFull Analysis: Found {len(eigenvals_valid_1)} valid modes (core + extended).")

    if len(eigenvals_valid_1) > 0:
        print(f"Eigenvalues (basis 1): {[f'{ev:.4f}' for ev in eigenvals_valid_1[:10]]}")
    if len(eigenvals_valid_2) > 0:
        print(f"Eigenvalues (basis 2): {[f'{ev:.4f}' for ev in eigenvals_valid_2[:10]]}")

    # Compute coefficients
    coefficients_1_dict = {}
    coefficients_2_dict = {}

    for pert_type in perturbation_types:
        if pert_type in transform_1_dict and len(eigenvals_valid_1) > 0:
            coefficients_1_dict[pert_type] = compute_coefficients(
                eigenvals_valid_1, eigenvecs_valid_1, transform_1_dict[pert_type])
            coefficients_2_dict[pert_type] = compute_coefficients(
                eigenvals_valid_2, eigenvecs_valid_2, transform_2_dict[pert_type])

    return {
        'eta_grid': eta_grid,
        'eigenvals_1': eigenvals_valid_1,
        'eigenvals_2': eigenvals_valid_2,
        'eigenvecs_1': eigenvecs_valid_1,
        'eigenvecs_2': eigenvecs_valid_2,
        'coefficients_1': coefficients_1_dict,
        'coefficients_2': coefficients_2_dict,
        'basis_1': basis_1_dict,
        'basis_2': basis_2_dict,
        'M_combined': M_combined
    }


def multi_perturbation_extension_reverse(N=23, N_t=500, folder_path='./data/', eigenvalues_threshold=0.99):
    """
    Same as multi_perturbation_analysis_with_full_extension but uses
    REVERSED column ordering before QR decomposition.

    Key insight (QR = Gram-Schmidt on columns, left to right):
      Forward ordering  [core | extend | highK_chunked]:
        high-k QR vectors = high-k raw - projections onto core QR vectors
        → core is different in basis_1 vs basis_2, so the projections differ
        → M[high-k block] ≠ I  →  missing high-k modes after Varimax

      Reversed ordering  [extend | highK_chunked | core]:
        high-k columns are processed FIRST with no prior QR vectors
        → same input in both bases → same QR output in both bases
        → M[high-k block] = I  (guaranteed, not approximate)
        → cross-terms M[high-k, core] = 0  (by GS orthogonality)

    The returned basis_1/basis_2 and coefficient dicts use the same
    reversed column ordering, so coefficient-based reconstruction is
    consistent.
    """
    print(f"Starting multi_perturbation_extension_reverse with N={N}, N_t={N_t}")

    eta_grid = np.linspace(0, fcb_time, N_t)
    print(f"Using eta_grid from cutoff to FCB: eta ∈ [0, {fcb_time:.4e}]")

    # ── Generate CORE bases (kept separate until concatenation) ───────────────
    print("\nGenerating core basis 1 (Closed Universe / integerK)...")
    basis_1_core = generate_multi_perturbation_bases("integerK", eta_grid, folder_path=folder_path)

    print("\nGenerating core basis 2 (Palindromic Universe / allowedK)...")
    basis_2_core = generate_multi_perturbation_bases("allowedK", eta_grid, folder_path=folder_path)

    # ── Collect ALL high-k extension data into a single combined dict ─────────
    # These columns are identical in both bases.  They must be the FIRST
    # columns passed to QR so they are orthogonalised before any
    # basis-specific core vectors can contaminate them.
    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    basis_highk_combined = {pert: None for pert in perturbation_types}
    N_highk_total = 0

    # (1) extend_integerK
    extend_data_path = folder_path + 'data_extend_integerK/'
    if os.path.exists(extend_data_path):
        print("\nLoading extend_integerK modes...")
        try:
            basis_extend_dict = generate_multi_perturbation_bases(
                "extend_integerK", eta_grid, folder_path=folder_path)
            for pert_type in perturbation_types:
                if pert_type in basis_extend_dict:
                    arr = basis_extend_dict[pert_type]
                    if basis_highk_combined[pert_type] is None:
                        basis_highk_combined[pert_type] = arr
                    else:
                        basis_highk_combined[pert_type] = np.concatenate(
                            [basis_highk_combined[pert_type], arr], axis=1)
            ref = next(v for v in basis_extend_dict.values() if v is not None)
            n_ext = ref.shape[1]
            N_highk_total += n_ext
            print(f"  extend_integerK: {n_ext} modes per perturbation type")
        except FileNotFoundError as e:
            print(f"  Warning: extend_integerK data not found: {e}")
    else:
        print(f"\nextend_integerK directory not found: {extend_data_path}")

    # (2) high-K chunked data
    # basis_highk_dict = load_and_merge_chunked_highk_data(folder_path, eta_grid)
    # if basis_highk_dict is not None:
    #     print("\nLoading high-K chunked modes...")
    #     for pert_type in perturbation_types:
    #         if pert_type in basis_highk_dict:
    #             arr = basis_highk_dict[pert_type]
    #             if basis_highk_combined[pert_type] is None:
    #                 basis_highk_combined[pert_type] = arr
    #             else:
    #                 basis_highk_combined[pert_type] = np.concatenate(
    #                     [basis_highk_combined[pert_type], arr], axis=1)
    #     ref = next(v for v in basis_highk_dict.values() if v is not None)
    #     n_chk = ref.shape[1]
    #     N_highk_total += n_chk
    #     print(f"  high-K chunked: {n_chk} modes per perturbation type")

    # ── Reversed-order concatenation: [highK_combined | core] ─────────────────
    print(f"\nBuilding reversed-order bases "
          f"[highK({N_highk_total}) | core] for QR...")
    basis_1_dict = {}
    basis_2_dict = {}
    for pert_type in perturbation_types:
        hk = basis_highk_combined.get(pert_type)
        c1 = basis_1_core.get(pert_type)
        c2 = basis_2_core.get(pert_type)
        if hk is not None and c1 is not None:
            basis_1_dict[pert_type] = np.concatenate([hk, c1], axis=1)
            basis_2_dict[pert_type] = np.concatenate([hk, c2], axis=1)
            print(f"  {pert_type}: [{hk.shape[1]} highK | {c1.shape[1]} core]  "
                  f"total = {basis_1_dict[pert_type].shape[1]}")
        elif c1 is not None:
            basis_1_dict[pert_type] = c1
            basis_2_dict[pert_type] = c2
            print(f"  {pert_type}: no high-K data found, using core only  "
                  f"shape = {c1.shape}")

    # ── QR decomposition on reversed-order full bases ─────────────────────────
    print("\n--- Running Eigen-Analysis on reversed-order full bases ---")
    ortho_funcs_1_dict = {}
    ortho_funcs_2_dict = {}
    transform_1_dict = {}
    transform_2_dict = {}

    for pert_type in perturbation_types:
        if pert_type not in basis_1_dict:
            continue
        print(f"  QR decomposition for {pert_type}")
        if np.all(basis_1_dict[pert_type] == 0):
            print(f"    Warning: basis_1 for {pert_type} is all zeros, skipping")
            continue
        if np.all(basis_2_dict[pert_type] == 0):
            print(f"    Warning: basis_2 for {pert_type} is all zeros, skipping")
            continue
        try:
            ortho_funcs_1_dict[pert_type], transform_1_dict[pert_type] = \
                qr_decomposition(basis_1_dict[pert_type])
            ortho_funcs_2_dict[pert_type], transform_2_dict[pert_type] = \
                qr_decomposition(basis_2_dict[pert_type])
            print(f"    Successfully processed {pert_type}")
        except Exception as e:
            print(f"    Error in QR decomposition for {pert_type}: {e}")

    # ── Eigenvalue analysis ────────────────────────────────────────────────────
    print("\nComputing combined eigenvalue analysis (reversed-order full basis)...")
    eigenvals_1, eigenvecs_1, eigenvals_2, eigenvecs_2, M_combined = \
        compute_multi_perturbation_A_matrix(ortho_funcs_1_dict, ortho_funcs_2_dict)

    eigenvals_valid_1, eigenvecs_valid_1 = choose_eigenvalues(
        eigenvals_1, eigenvecs_1, eigenvalues_threshold)
    eigenvals_valid_2, eigenvecs_valid_2 = choose_eigenvalues(
        eigenvals_2, eigenvecs_2, eigenvalues_threshold)

    print(f"\nFound {len(eigenvals_valid_1)} valid modes (reversed-order full basis).")
    if len(eigenvals_valid_1) > 0:
        print(f"Eigenvalues (basis 1): {[f'{ev:.4f}' for ev in eigenvals_valid_1[:10]]}")
    if len(eigenvals_valid_2) > 0:
        print(f"Eigenvalues (basis 2): {[f'{ev:.4f}' for ev in eigenvals_valid_2[:10]]}")

    # ── Coefficients ──────────────────────────────────────────────────────────
    # Coefficient columns follow the reversed column ordering:
    #   cols 0..N_highk_total-1  → high-K extension modes
    #   cols N_highk_total..end  → core modes
    # basis_1_dict / basis_2_dict use the same ordering, so reconstruction
    #   solution = sum_j coeff[j] * basis[pert][:, j]  is consistent.
    coefficients_1_dict = {}
    coefficients_2_dict = {}
    for pert_type in perturbation_types:
        if pert_type in transform_1_dict and len(eigenvals_valid_1) > 0:
            coefficients_1_dict[pert_type] = compute_coefficients(
                eigenvals_valid_1, eigenvecs_valid_1, transform_1_dict[pert_type])
            coefficients_2_dict[pert_type] = compute_coefficients(
                eigenvals_valid_2, eigenvecs_valid_2, transform_2_dict[pert_type])

    return {
        'eta_grid': eta_grid,
        'eigenvals_1': eigenvals_valid_1,
        'eigenvals_2': eigenvals_valid_2,
        'eigenvecs_1': eigenvecs_valid_1,
        'eigenvecs_2': eigenvecs_valid_2,
        'coefficients_1': coefficients_1_dict,
        'coefficients_2': coefficients_2_dict,
        'basis_1': basis_1_dict,
        'basis_2': basis_2_dict,
        'M_combined': M_combined,
        'N_highk': N_highk_total,   # number of high-k columns prepended
    }


def multi_perturbation_analysis(N=23, N_t=500, folder_path='./data/', eigenvalues_threshold=0.99):
    """
    Complete multi-perturbation eigenvalue analysis with divide-and-conquer for high-k extension.

    This uses the "divide and conquer" approach:
    1. Run eigenvalue analysis ONLY on low-k (core) modes
    2. Manually append high-k extension as block-diagonal identity matrix
    3. This prevents artificial high-k eigenvalues (1.0) from dominating the low-k physics
    """
    print(f"Starting multi-perturbation analysis with N={N}, N_t={N_t}")

    # Define eta_grid from cutoff_time to fcb_time as requested
    eta_grid = np.linspace(0, fcb_time, N_t)
    print(f"Using eta_grid from cutoff to FCB: eta ∈ [0, {fcb_time:.4e}]")

    # 1. Generate ONLY the Low-K (Physics) Bases first
    print("\nGenerating CORE basis 1 (Closed Universe)...")
    basis_1_dict = generate_multi_perturbation_bases("integerK", eta_grid, folder_path=folder_path)

    print("\nGenerating CORE basis 2 (Palindromic Universe)...")
    basis_2_dict = generate_multi_perturbation_bases("allowedK", eta_grid, folder_path=folder_path)

    # 2. Check for Extended Basis, but DO NOT merge yet
    basis_extend_dict = None
    extend_data_path = folder_path + 'data_extend_integerK/'
    if os.path.exists(extend_data_path):
        print("\nLoading extended basis (to be appended later)...")
        try:
            basis_extend_dict = generate_multi_perturbation_bases("extend_integerK", eta_grid, folder_path=folder_path)
            print(f"  Extended basis loaded successfully")
        except FileNotFoundError as e:
            print(f"  Warning: Extended basis data files not found: {e}")
            basis_extend_dict = None
    else:
        print(f"\nExtended basis directory not found: {extend_data_path}")
        print("Continuing with core bases only...")

    # 3. Perform Analysis ONLY on the Core (Low-K)
    # This ensures the solver focuses on the mismatch region
    print("\n--- Running Eigen-Analysis on Core Low-K Modes ---")

    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    ortho_funcs_1_dict = {}
    ortho_funcs_2_dict = {}
    transform_1_dict = {}
    transform_2_dict = {}

    for pert_type in perturbation_types:
        if pert_type not in basis_1_dict:
            continue

        print(f"  QR decomposition for {pert_type} (core only)")

        # Check for valid data
        if np.all(basis_1_dict[pert_type] == 0):
            print(f"    Warning: Basis 1 for {pert_type} is all zeros, skipping")
            continue
        if np.all(basis_2_dict[pert_type] == 0):
            print(f"    Warning: Basis 2 for {pert_type} is all zeros, skipping")
            continue

        try:
            # QR on core only
            ortho_funcs_1_dict[pert_type], transform_1_dict[pert_type] = qr_decomposition(basis_1_dict[pert_type])
            ortho_funcs_2_dict[pert_type], transform_2_dict[pert_type] = qr_decomposition(basis_2_dict[pert_type])
            print(f"    Successfully processed {pert_type}")
        except Exception as e:
            print(f"    Error in QR decomposition for {pert_type}: {e}")

    # Compute A matrix and Eigenvalues on CORE
    print("\nComputing combined eigenvalue analysis (core only)...")
    eigenvals_1, eigenvecs_1, eigenvals_2, eigenvecs_2, M_combined = compute_multi_perturbation_A_matrix(
        ortho_funcs_1_dict, ortho_funcs_2_dict)

    # Filter valid eigenvalues from CORE
    eigenvals_valid_1, eigenvecs_valid_1 = choose_eigenvalues(
        eigenvals_1, eigenvecs_1, eigenvalues_threshold)
    eigenvals_valid_2, eigenvecs_valid_2 = choose_eigenvalues(
        eigenvals_2, eigenvecs_2, eigenvalues_threshold)

    print(f"\nCore Analysis: Found {len(eigenvals_valid_1)} valid modes in mismatch region.")

    if len(eigenvals_valid_1) > 0:
        print(f"Core Eigenvalues (basis 1): {[f'{ev:.4f}' for ev in eigenvals_valid_1[:5]]}")
    if len(eigenvals_valid_2) > 0:
        print(f"Core Eigenvalues (basis 2): {[f'{ev:.4f}' for ev in eigenvals_valid_2[:5]]}")

    # 4. Compute Coefficients for Core
    coefficients_1_dict = {}
    coefficients_2_dict = {}

    for pert_type in perturbation_types:
        if pert_type in transform_1_dict and len(eigenvals_valid_1) > 0:
            coefficients_1_dict[pert_type] = compute_coefficients(
                eigenvals_valid_1, eigenvecs_valid_1, transform_1_dict[pert_type])
            coefficients_2_dict[pert_type] = compute_coefficients(
                eigenvals_valid_2, eigenvecs_valid_2, transform_2_dict[pert_type])

    # 5. MERGE: Manually append the High-K extension now
    # The high-k modes are diagonal (Identity) because Basis 1 == Basis 2
    if basis_extend_dict is not None:
        print("\n--- Appending Extended High-K Modes ---")

        # Number of extended modes
        N_ext = basis_extend_dict['vr'].shape[1]
        print(f"Appending {N_ext} high-k modes as block-diagonal identity")

        # Get the dimension of the QR-orthonormal space from core analysis
        # This is the number of rows in the M_combined matrix
        N_core = M_combined.shape[0]

        # 1. Extend Eigenvalues (Assume perfect 1.0 for extension)
        ext_evals = np.ones(N_ext)
        eigenvals_valid_1 = np.concatenate([eigenvals_valid_1, ext_evals])
        eigenvals_valid_2 = np.concatenate([eigenvals_valid_2, ext_evals])

        # 2. Extend Eigenvectors
        # The QR space now has dimension N_core + N_ext
        # We need to:
        # a) Pad existing core eigenvectors with zeros
        # b) Add new eigenvectors for the extended modes (standard basis vectors)

        # First, extend all existing core eigenvectors by padding with zeros
        extended_eigenvecs_1 = []
        extended_eigenvecs_2 = []

        for eigvec in eigenvecs_valid_1:
            # Pad with zeros to extend to N_core + N_ext dimensions
            extended = np.zeros(N_core + N_ext)
            extended[:N_core] = eigvec
            extended_eigenvecs_1.append(extended)

        for eigvec in eigenvecs_valid_2:
            # Pad with zeros to extend to N_core + N_ext dimensions
            extended = np.zeros(N_core + N_ext)
            extended[:N_core] = eigvec
            extended_eigenvecs_2.append(extended)

        # Now add the extended high-k mode eigenvectors (standard basis vectors)
        for i in range(N_ext):
            # Create a zero vector in the extended space
            ext_eigenvec = np.zeros(N_core + N_ext)
            # Set the corresponding extended dimension to 1
            ext_eigenvec[N_core + i] = 1.0

            # Append to both bases (they're identical for high-k)
            extended_eigenvecs_1.append(ext_eigenvec)
            extended_eigenvecs_2.append(ext_eigenvec)

        # Replace the original eigenvector lists
        eigenvecs_valid_1 = extended_eigenvecs_1
        eigenvecs_valid_2 = extended_eigenvecs_2

        print(f"Extended eigenvector lists: now {len(eigenvecs_valid_1)} eigenvectors in {N_core + N_ext}-dimensional space")

        # 3. Extend M_combined matrix
        # M_combined represents the mixing between bases
        # For the extended high-k modes, there's no mixing (Basis 1 == Basis 2)
        # So we create a block diagonal: [ M_core   0   ]
        #                                 [   0     I   ]
        M_extended = np.zeros((N_core + N_ext, N_core + N_ext))
        M_extended[:N_core, :N_core] = M_combined
        M_extended[N_core:, N_core:] = np.eye(N_ext)
        M_combined = M_extended

        print(f"Extended M_combined matrix from {N_core}x{N_core} to {N_core + N_ext}x{N_core + N_ext}")

        # 4. Extend Coefficients
        # For the extension, create block diagonal structure for coefficients

        for pert_type in perturbation_types:
            if pert_type in coefficients_1_dict and pert_type in basis_extend_dict:
                core_coeffs = coefficients_1_dict[pert_type]

                # Create the extended coefficient matrix
                # Structure: [ Core_Coeffs   0 ]
                #            [ 0             I ]

                # Current shape: (K_core, N_core)
                K_core, N_core = core_coeffs.shape

                # New shape: (K_core + N_ext, N_core + N_ext)
                new_coeffs = np.zeros((K_core + N_ext, N_core + N_ext))

                # Fill top-left with core results
                new_coeffs[:K_core, :N_core] = core_coeffs

                # Fill bottom-right with Identity (high-k maps to high-k)
                new_coeffs[K_core:, N_core:] = np.eye(N_ext)

                coefficients_1_dict[pert_type] = new_coeffs

                # Repeat for Basis 2 (assuming identical extension)
                core_coeffs_2 = coefficients_2_dict[pert_type]
                new_coeffs_2 = np.zeros((K_core + N_ext, N_core + N_ext))
                new_coeffs_2[:K_core, :N_core] = core_coeffs_2
                new_coeffs_2[K_core:, N_core:] = np.eye(N_ext)
                coefficients_2_dict[pert_type] = new_coeffs_2

                # Update the basis dicts for plotting later
                basis_1_dict[pert_type] = np.concatenate([basis_1_dict[pert_type], basis_extend_dict[pert_type]], axis=1)
                basis_2_dict[pert_type] = np.concatenate([basis_2_dict[pert_type], basis_extend_dict[pert_type]], axis=1)

                print(f"  Extended {pert_type}: coeffs shape = {new_coeffs.shape}, basis shape = {basis_1_dict[pert_type].shape}")

        print(f"Total modes after extension: {len(eigenvals_valid_1)} (core: {K_core}, extended: {N_ext})")

    # # save the coefficients
    # with open("multi_perturbation_coefficients1.pickle", 'wb') as f:
    #     pickle.dump(coefficients_1_dict, f)

    return {
        'eta_grid': eta_grid,
        'eigenvals_1': eigenvals_valid_1,
        'eigenvals_2': eigenvals_valid_2,
        'eigenvecs_1': eigenvecs_valid_1,
        'eigenvecs_2': eigenvecs_valid_2,
        'coefficients_1': coefficients_1_dict,
        'coefficients_2': coefficients_2_dict,
        'basis_1': basis_1_dict,
        'basis_2': basis_2_dict,
        'M_combined': M_combined
    }

def plot_multi_perturbation_results(results, N_plot=3):
    """
    Plot the combined eigenfunctions from multi-perturbation analysis
    """
    eta_grid = results['eta_grid']
    eigenvals_1 = results['eigenvals_1']
    coefficients_1 = results['coefficients_1']
    coefficients_2 = results['coefficients_2']
    basis_1 = results['basis_1']
    basis_2 = results['basis_2']
    
    N_plot = min(N_plot, len(eigenvals_1))
    if N_plot == 0:
        print("No valid eigenfunctions to plot")
        return
    
    perturbation_types = ['dr', 'dm', 'vr', 'vm']
    n_pert = len(perturbation_types)
    
    fig, axes = plt.subplots(N_plot, n_pert, figsize=(16, 2*N_plot), 
                           constrained_layout=True)
    if N_plot == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Multi-Perturbation Common Eigenfunctions", fontsize=16)
    
    for i in range(N_plot):
        for j, pert_type in enumerate(perturbation_types):
            ax = axes[i, j]
            
            if pert_type not in coefficients_1 or pert_type not in coefficients_2:
                ax.text(0.5, 0.5, f"No data\nfor {pert_type}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{pert_type} (eigenval {i+1})")
                continue
            
            # Reconstruct solutions from both bases
            solution_1 = np.zeros_like(eta_grid)
            solution_2 = np.zeros_like(eta_grid)
            
            N = len(coefficients_1[pert_type][i])
            for k in range(N):
                solution_1 += coefficients_1[pert_type][i, k] * basis_1[pert_type][:, k]
                solution_2 += coefficients_2[pert_type][i, k] * basis_2[pert_type][:, k]
            
            # Sign alignment
            if np.dot(solution_1, solution_2) < 0:
                solution_2 *= -1
            
            # Plot
            ax.plot(eta_grid, solution_1, 'r-', linewidth=2.5, 
                   label='Basis 1 (Closed)', alpha=0.8)
            ax.plot(eta_grid, solution_2, 'g--', linewidth=2.0, 
                   label='Basis 2 (Palindromic)', alpha=0.8)
            
            ax.set_title(f"{pert_type} (λ={eigenvals_1[i]:.3f})")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
            if i == N_plot - 1:
                ax.set_xlabel("Conformal Time η")

    plt.savefig("multi_perturbation_eigenfunctions_after_recombination.pdf", dpi=300, bbox_inches='tight')

def plot_coefficients_by_dominant_k(results, eigenvalue_threshold=0.95, N_plot=10):
    """
    Plot coefficients of eigenfunctions sorted by dominant k mode

    Parameters:
    -----------
    results : dict
        Results dictionary from multi_perturbation_analysis
    eigenvalue_threshold : float
        Minimum eigenvalue to include (default: 0.95)
    N_plot : int
        Maximum number of eigenfunctions to plot
    """
    eigenvals_1 = np.array(results['eigenvals_1'])
    eigenvecs_1 = results['eigenvecs_1']

    # Step 1: Filter eigenvectors with eigenvalues > threshold
    valid_mask = eigenvals_1.real > eigenvalue_threshold
    eigenvals_valid = eigenvals_1[valid_mask]

    print(f"\nFiltering eigenvectors with eigenvalue > {eigenvalue_threshold}")
    print(f"Found {len(eigenvals_valid)} eigenvectors above threshold out of {len(eigenvals_1)} total")

    if len(eigenvals_valid) == 0:
        print("No eigenvectors found above threshold. Try lowering the threshold.")
        return

    # Get valid indices
    valid_indices = np.where(valid_mask)[0]

    # Step 2: Sort by dominant k mode
    # Extract coefficients for the reference perturbation
    # Convert list of eigenvectors to numpy array first
    eigenvecs_1_array = np.array([eigenvecs_1[i] for i in range(len(eigenvecs_1))])
    coeffs_ref = eigenvecs_1_array[valid_indices]

    # Find the index of the maximum coefficient for each eigenfunction (dominant k)
    max_indices = np.argmax(np.abs(coeffs_ref.real), axis=1)

    # Sort eigenfunctions based on the dominant k mode (from small to large)
    sorted_order = np.argsort(max_indices)

    print(f"Dominant k indices: {max_indices[sorted_order]}")
    print(f"Corresponding eigenvalues: {eigenvals_valid[sorted_order].real}")

    # Step 3: Create bar chart plots
    N_plot = min(N_plot, len(eigenvals_valid))
    N_basis = coeffs_ref.shape[1]  # Number of basis functions

    # Create figure with subplots
    fig, axs = plt.subplots(N_plot, figsize=(3.8, 0.7*N_plot))
    if N_plot == 1:
        axs = [axs]

    fig.suptitle(f"Coefficients sorted by dominant k", fontsize=10)

    # Plot each eigenfunction's coefficients
    for i in range(N_plot):
        plot_idx = sorted_order[i]  # Index in the filtered arrays
        global_idx = valid_indices[plot_idx]  # Index in the original arrays

        coefficients = coeffs_ref[plot_idx, :].real
        dominant_k = max_indices[plot_idx]
        eigenval = eigenvals_valid[plot_idx].real

        # Create bar plot
        k_values = np.arange(1, N_basis + 1)
        axs[i].bar(k_values, coefficients, width=0.4)
        axs[i].set_xlim(0, min(20, N_basis + 1))
        axs[i].set_ylim(-1, 1.)

        # Add title with dominant k and eigenvalue
        axs[i].set_title(f"λ={eigenval:.3f}, dominant k={dominant_k+1}",
                        fontsize=8, loc='right')

        # Highlight the dominant k
        axs[i].axvline(dominant_k + 1, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)

    # Format axes
    for ax in axs:
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.label_outer()
        ax.grid(True, alpha=0.3, axis='y')

    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[-1].set_xlabel(r"$k$ mode index", fontsize=8)
    fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

    fig.tight_layout()

    # Save figure
    output_filename = f"multi_perturbation_coefficients_sorted.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved coefficient plot to {output_filename}")
    plt.show()

    return {
        'valid_indices': valid_indices,
        'sorted_order': sorted_order,
        'dominant_k': max_indices,
        'eigenvals_valid': eigenvals_valid
    }


# def plot_coefficients_by_dominant_k_pert_type(results, eigenvalue_threshold=0.95, N_plot=10,
#                                     reference_perturbation='vr'):
#     """
#     Plot coefficients of eigenfunctions sorted by dominant k mode

#     Parameters:
#     -----------
#     results : dict
#         Results dictionary from multi_perturbation_analysis
#     eigenvalue_threshold : float
#         Minimum eigenvalue to include (default: 0.95)
#     N_plot : int
#         Maximum number of eigenfunctions to plot
#     reference_perturbation : str
#         Which perturbation type to use for determining dominant k (default: 'vr')
#     """
#     eigenvals_1 = np.array(results['eigenvals_1'])
#     coefficients_1 = results['coefficients_1']

#     if reference_perturbation not in coefficients_1:
#         print(f"Error: {reference_perturbation} not found in coefficients")
#         print(f"Available perturbation types: {list(coefficients_1.keys())}")
#         return

#     # Step 1: Filter eigenvectors with eigenvalues > threshold
#     valid_mask = eigenvals_1.real > eigenvalue_threshold
#     eigenvals_valid = eigenvals_1[valid_mask]

#     print(f"\nFiltering eigenvectors with eigenvalue > {eigenvalue_threshold}")
#     print(f"Found {len(eigenvals_valid)} eigenvectors above threshold out of {len(eigenvals_1)} total")

#     if len(eigenvals_valid) == 0:
#         print("No eigenvectors found above threshold. Try lowering the threshold.")
#         return

#     # Get valid indices
#     valid_indices = np.where(valid_mask)[0]

#     # Step 2: Sort by dominant k mode
#     # Extract coefficients for the reference perturbation
#     coeffs_ref = coefficients_1[reference_perturbation][valid_indices]

#     # Find the index of the maximum coefficient for each eigenfunction (dominant k)
#     max_indices = np.argmax(np.abs(coeffs_ref.real), axis=1)

#     # Sort eigenfunctions based on the dominant k mode (from small to large)
#     sorted_order = np.argsort(max_indices)

#     print(f"Dominant k indices: {max_indices[sorted_order]}")
#     print(f"Corresponding eigenvalues: {eigenvals_valid[sorted_order].real}")

#     # Step 3: Create bar chart plots
#     N_plot = min(N_plot, len(eigenvals_valid))
#     N_basis = coeffs_ref.shape[1]  # Number of basis functions

#     # Create figure with subplots
#     fig, axs = plt.subplots(N_plot, figsize=(3.8, 0.7*N_plot))
#     if N_plot == 1:
#         axs = [axs]

#     fig.suptitle(f"Coefficients sorted by dominant k ({reference_perturbation})", fontsize=10)

#     # Plot each eigenfunction's coefficients
#     for i in range(N_plot):
#         plot_idx = sorted_order[i]  # Index in the filtered arrays
#         global_idx = valid_indices[plot_idx]  # Index in the original arrays

#         coefficients = coeffs_ref[plot_idx, :].real
#         dominant_k = max_indices[plot_idx]
#         eigenval = eigenvals_valid[plot_idx].real

#         # Create bar plot
#         k_values = np.arange(1, N_basis + 1)
#         axs[i].bar(k_values, coefficients, width=0.4)
#         axs[i].set_xlim(0, min(20, N_basis + 1))
#         axs[i].set_ylim(-1, 1.)

#         # Add title with dominant k and eigenvalue
#         axs[i].set_title(f"λ={eigenval:.3f}, dominant k={dominant_k+1}",
#                         fontsize=8, loc='right')

#         # Highlight the dominant k
#         axs[i].axvline(dominant_k + 1, color='red', linestyle='--',
#                       alpha=0.5, linewidth=1)

#     # Format axes
#     for ax in axs:
#         ax.xaxis.set_tick_params(labelsize=8)
#         ax.yaxis.set_tick_params(labelsize=8)
#         ax.label_outer()
#         ax.grid(True, alpha=0.3, axis='y')

#     axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
#     axs[-1].set_xlabel(r"$k$ mode index", fontsize=8)
#     fig.text(0.02, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=9)

#     fig.tight_layout()

#     # Save figure
#     output_filename = f"multi_perturbation_coefficients_{reference_perturbation}_sorted.pdf"
#     plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#     print(f"\nSaved coefficient plot to {output_filename}")
#     plt.show()

#     return {
#         'valid_indices': valid_indices,
#         'sorted_order': sorted_order,
#         'dominant_k': max_indices,
#         'eigenvals_valid': eigenvals_valid
#     }

if __name__ == "__main__":
    # Run the analysis
    folder_path = f'./data/'
    allowedK = np.load(folder_path + 'data_allowedK/L70_kvalues.npy')
    results = multi_perturbation_analysis(N=len(allowedK), N_t=1000)

    # Plot results
    if len(results['eigenvals_1']) > 0:
        plot_multi_perturbation_results(results, N_plot=5)

        # Plot coefficients sorted by dominant k mode
        print("\n" + "="*80)
        print("Plotting coefficients sorted by dominant k mode")
        print("="*80)
        sorting_results = plot_coefficients_by_dominant_k(
            results,
            eigenvalue_threshold=0.95,
            N_plot=6
        )
    else:
        print("No eigenvalues found for plotting")

    print("\nAnalysis complete!")
    print(f"Combined M matrix shape: {results['M_combined'].shape}")
    print(f"Number of common eigenfunctions found: {len(results['eigenvals_1'])}")