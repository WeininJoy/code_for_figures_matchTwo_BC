# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:03:58 2021
Modified to save the full time evolution of the transfer matrix basis vectors.
PARALLELIZED VERSION using multiprocessing.

This script pre-calculates the time-dependent transfer matrices U(t, t_prime)
by integrating basis vectors from a point near the FCB (t_prime = endtime)
back to recombination and saving the full solution history for each.

@author: MRose (Modified by AI, Parallelized)
"""

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

def process_single_k_timeseries(k, s_init, t_grid_a, t_grid_s, num_time_points,
                                 endtime, swaptime, recConformalTime,
                                 H0, OmegaLambda, OmegaM, OmegaR, OmegaK,
                                 deltaeta_max, k_deltaeta_target,
                                 atol, rtol, num_variables):
    """
    Process a single k value to compute time series solutions for ABC, DEF, and GHI.

    This function is designed to be called in parallel for different k values.

    Returns:
    --------
    tuple: (ABC_sols_k, DEF_sols_k, GHI_sols_k)
        - ABC_sols_k: shape (num_variables, 6, num_time_points)
        - DEF_sols_k: shape (num_variables, 2, num_time_points)
        - GHI_sols_k: shape (num_variables, num_time_points)
    """

    # ADAPTIVE DELTAETA: Compute k-specific deltaeta for boundary conditions
    # deltaeta = min(k_deltaeta_target / k, deltaeta_max)
    deltaeta = deltaeta_max # set it as deltaeta_max for now, can be adjusted based on k if needed

    print(f"Processing k = {k:.6f}, deltaeta = {deltaeta:.6e}, k*deltaeta = {k*deltaeta:.6f}")

    # Define derivative functions for this k value
    def dX2_dt(t, X):
        s, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
        sdot = -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

        rho_m = 3*(H0**2)*OmegaM*(abs(s)**3)
        rho_r = 3*(H0**2)*OmegaR*(abs(s)**4)

        phidot = (sdot/s)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*s**2)
        fr2dot = -(8/15)*(k**2)*vr - 0.6*k*X[8]
        psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*s)*(sdot*fr2 + 0.5*s*fr2dot)
        drdot = (4/3)*(3*phidot + (k**2)*vr)
        dmdot = 3*phidot + vm*(k**2)
        vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
        vmdot = (sdot/s)*vm - psi
        derivatives = [sdot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

        for j in range(8, num_variables):
            l = j - 5
            derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

        lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
        derivatives.append(lastderiv)
        return derivatives

    def dX3_dt(t, X):
        sigma, phi, psi, dr, dm, vr, vm, fr2 = X[0:8]
        sigmadot = -(H0)*np.sqrt((OmegaLambda*np.exp(-2*sigma)+OmegaK+OmegaM*np.exp(sigma)
                                +OmegaR*np.exp(2*sigma)))

        rho_m = 3*(H0**2)*OmegaM*(np.exp(3*sigma))
        rho_r = 3*(H0**2)*OmegaR*(np.exp(4*sigma))

        phidot = (sigmadot)*psi - ((4/3)*rho_r*vr + rho_m*vm)/(2*np.exp(2*sigma))
        fr2dot = -(8/15)*(k**2)*vr - (3/5)*k*X[8]
        psidot = phidot - (1/k**2)*(6*(H0**2)*OmegaR*np.exp(sigma))*(sigmadot*np.exp(sigma)*fr2 + 0.5*np.exp(sigma)*fr2dot)
        drdot = (4/3)*(3*phidot + (k**2)*vr)
        dmdot = 3*phidot + vm*(k**2)
        vrdot = -(psi + dr/4) + (1 + 3*OmegaK*H0**2/k**2)*fr2/2
        vmdot = (sigmadot)*vm - psi
        derivatives = [sigmadot, phidot, psidot, drdot, dmdot, vrdot, vmdot, fr2dot]

        for j in range(8, num_variables):
            l = j - 5
            derivatives.append((k/(2*l+1))*(l*X[j-1] - (l+1)*X[j+1]))

        lastderiv = k*X[num_variables-1] - ((num_variables-5 + 1)*X[num_variables])/t
        derivatives.append(lastderiv)
        return derivatives

    # Initialize tensors to store solutions for this k
    ABC_sols_k = np.zeros((num_variables, 6, num_time_points))
    DEF_sols_k = np.zeros((num_variables, 2, num_time_points))
    GHI_sols_k = np.zeros((num_variables, num_time_points))

    # --- A: Calculate solutions for base variable basis vectors (forms ABC) ---
    for n in range(6):
        x0 = np.zeros(num_variables)
        x0[n] = 1
        inits_s = np.concatenate(([s_init], x0))

        # Integrate from endtime to swaptime (s-domain)
        sol_s = solve_ivp(dX2_dt, [endtime, swaptime], inits_s, dense_output=True,
                        method='LSODA', atol=atol, rtol=rtol)

        # Prepare for next integration leg
        inits_a = sol_s.y[:, -1]
        inits_a[0] = 1./inits_a[0]

        # Integrate from swaptime to recombination (sigma-domain)
        sol_a = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits_a, dense_output=True,
                            method='LSODA', atol=atol, rtol=rtol)

        # Evaluate solutions on the grid and stitch them
        sol_on_s_grid = sol_s.sol(t_grid_s)
        sol_on_a_grid = sol_a.sol(t_grid_a)

        # Convert a back to s for consistency
        sol_on_a_grid[0, :] = 1./sol_on_a_grid[0, :]

        # Stitch the solutions together
        full_solution = np.concatenate((sol_on_a_grid, sol_on_s_grid), axis=1)

        # Store the perturbation part (ignoring the background variable)
        ABC_sols_k[:, n, :] = full_solution[1:, :]

    # --- B: Calculate solutions for anisotropic basis vectors (forms DEF) ---
    for j in range(2):
        x0 = np.zeros(num_variables)
        x0[j+7] = 1  # Initial conditions for F_2 and F_3
        inits_s = np.concatenate(([s_init], x0))

        sol_s = solve_ivp(dX2_dt, [endtime, swaptime], inits_s, dense_output=True,
                        method='LSODA', atol=atol, rtol=rtol)
        inits_a = sol_s.y[:, -1]
        inits_a[0] = 1/inits_a[0]
        sol_a = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits_a, dense_output=True,
                        method='LSODA', atol=atol, rtol=rtol)

        sol_on_s_grid = sol_s.sol(t_grid_s)
        sol_on_a_grid = sol_a.sol(t_grid_a)
        sol_on_a_grid[0, :] = 1./sol_on_a_grid[0, :]
        full_solution = np.concatenate((sol_on_a_grid, sol_on_s_grid), axis=1)
        DEF_sols_k[:, j, :] = full_solution[1:, :]

    # --- C: Calculate inhomogeneous solution part (forms GHI) ---
    x0 = np.zeros(num_variables)
    x3 = -(16/945) * (k**4) * (deltaeta**3)
    x0[9] = x3  # Inhomogeneous term from v_r^\infty
    inits_s = np.concatenate(([s_init], x0))

    sol_s = solve_ivp(dX2_dt, [endtime, swaptime], inits_s, dense_output=True,
                    method='LSODA', atol=atol, rtol=rtol)
    inits_a = sol_s.y[:, -1]
    inits_a[0] = 1./inits_a[0]
    sol_a = solve_ivp(dX3_dt, [swaptime, recConformalTime], inits_a, dense_output=True,
                    method='LSODA', atol=atol, rtol=rtol)

    sol_on_s_grid = sol_s.sol(t_grid_s)
    sol_on_a_grid = sol_a.sol(t_grid_a)
    sol_on_a_grid[0, :] = 1./sol_on_a_grid[0, :]
    full_solution = np.concatenate((sol_on_a_grid, sol_on_s_grid), axis=1)
    GHI_sols_k[:, :] = full_solution[1:, :]

    return (ABC_sols_k, DEF_sols_k, GHI_sols_k)


def compute_U_matrices_timeseries(params, z_rec, kvalues, folder_path, n_processes=None,
                                   num_variables=200,
                                   k_for_endtime=None, cosmo_param_bool=False):
    """
    Compute ABC/DEF/GHI time series solutions for perturbation analysis (PARALLELIZED).

    Parameters:
    -----------
    params : list or tuple
        [mt, kt, omega_b_ratio, h] cosmological parameters
    z_rec : float
        Recombination redshift
    kvalues : array-like
        Array of k values to process
    folder_path : str
        Path to save output files
    n_processes : int, optional
        Number of parallel processes to use. If None, uses all available CPU cores.
    num_variables : int, optional
        Number of perturbation variables used in the ODE solver (default: 200).
        Full arrays are saved; use larger values for higher k for better accuracy.
    k_for_endtime : float or None, optional
        The k value used to determine the adaptive endtime for the common time
        grid (endtime = fcb_time - k_deltaeta_target / k_for_endtime).
        Pass the global k_min across ALL k-range chunks so every chunk produces
        the same t_grid and their solutions can be concatenated without
        re-interpolation.  If None, defaults to np.min(kvalues).
    cosmo_param_bool : bool, optional
        If True, use cosmological parameters OmegaM, OmegaK, omega_b_ratio, h.
        If False, use mt, kt, omega_b_ratio, h. Default is False.

    Returns:
    --------
    None (saves results to disk)
    """

    # Constants
    lam = 1
    rt = 1
    Omega_gamma_h2 = 2.47e-5  # photon density
    Neff = 3.046
    
    if cosmo_param_bool == True:
        # Unpack parameters
        OmegaM, OmegaK, omega_b_ratio, h = params
        OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
        OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
    else:
        # Unpack parameters
        mt, kt, omega_b_ratio, h = params

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

        s0, OmegaLambda, OmegaM, OmegaK = cosmological_parameters(mt, kt, h)
        OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    ###############################################

    # Set tolerances
    atol = 1e-13
    rtol = 1e-13
    stol = 1e-10

    swaptime = 2  # set time when we swap from s to sigma

    # ADAPTIVE DELTAETA PARAMETERS
    deltaeta_max = 6.6e-4  # Original fixed value
    k_deltaeta_target = 0.005  # Ensures k*deltaeta < 0.005 for all k

    H0 = 1/np.sqrt(3*OmegaLambda)  # we are working in units of Lambda=c=1
    Hinf = H0*np.sqrt(OmegaLambda)

    # BACKGROUND EQUATIONS
    def ds_dt(t, s):
        return -1*H0*np.sqrt((OmegaLambda + OmegaK*abs(((s**2))) + OmegaM*abs(((s**3))) + OmegaR*abs((s**4))))

    t0 = 1e-8

    # Set coefficients for initial conditions
    smin1 = np.sqrt(3*OmegaLambda/OmegaR)
    szero = - OmegaM/(4*OmegaR)
    s1 = (OmegaM**2)/(16*np.sqrt(3*OmegaLambda*OmegaR**3)) - OmegaK/(6*np.sqrt(3*OmegaLambda*OmegaR))
    s2 = (OmegaM**3)/(192*OmegaLambda*OmegaR**2) + OmegaK*OmegaM/(48*OmegaLambda*OmegaR)
    s3 = (5*OmegaM**4 - 128*OmegaLambda*(OmegaR**3) -80./3.*OmegaM**2*OmegaR*OmegaK + 224./9.*OmegaR**2*OmegaK**2)/(3840*np.sqrt(3*(OmegaR**5)*(OmegaLambda**3)))
    s4 = (-OmegaM**5+20./3.*OmegaM**3*OmegaR*OmegaK - 32./3.*OmegaM*OmegaR**2*OmegaK**2)/(9216*(OmegaR**3)*(OmegaLambda**2))

    s0 = smin1/t0 + szero + s1*t0 + s2*t0**2 + s3*t0**3 + s4*t0**4

    print('Performing Initial Background Integration')
    def reach_FCB(t, s): return s[0]
    reach_FCB.terminal = True

    sol = solve_ivp(ds_dt, [t0, 12], [s0], max_step=0.25e-4, events=reach_FCB, method='LSODA', atol=atol, rtol=rtol)
    print('Initial Background Integration Done')

    # Check if t_events[0] is not empty before trying to access its elements
    if sol.t_events and len(sol.t_events[0]) > 0:
        fcb_time = sol.t_events[0][0]
        print(f"fcb_time: {fcb_time}")
    else:
        print(f"Event 'reach_FCB' did not occur.")
        fcb_time = None

    if fcb_time is not None:
        print(f"Further processing with fcb_time = {fcb_time}")
    else:
        print(f"No fcb_time available for further processing.")
        return

    # RECOMBINATION CONFORMAL TIME
    # Build continuous s -> t interpolant (s is monotonically decreasing; reverse for interp1d)
    _s_arr = sol.y[0][::-1]
    _t_arr = sol.t[::-1]
    _interp_t_from_s = interp1d(_s_arr, _t_arr, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')
    s_rec = 1 + z_rec  # reciprocal scale factor at recombination
    recConformalTime = float(_interp_t_from_s(s_rec))

    # ADAPTIVE DELTAETA: Compute endtime for the common time grid.
    # Use k_for_endtime if provided (pass the global k_min across all chunks so
    # every chunk shares the same t_grid and solutions can be concatenated).
    k_for_t_grid = k_for_endtime if k_for_endtime is not None else np.min(kvalues)
    deltaeta_for_grid = min(k_deltaeta_target / k_for_t_grid, deltaeta_max)
    endtime = fcb_time - deltaeta_for_grid
    print(f"Using endtime = {endtime} (fcb_time - {deltaeta_for_grid}) for time grid"
          f"  [k_for_endtime={k_for_t_grid:.6f}]")

    # Define a common time grid
    num_time_points = 500
    t_grid = np.linspace(recConformalTime, endtime, num=num_time_points)

    # Split grid for the two integration domains
    t_grid_a = t_grid[t_grid < swaptime]
    t_grid_s = t_grid[t_grid >= swaptime]

    # This is the starting value of s at `endtime`, used for all initial conditions
    s_init = np.interp(endtime, sol.t, sol.y[0])

    # Determine number of processes to use
    if n_processes is None:
        n_processes = cpu_count()

    print(f"\nUsing {n_processes} parallel processes to compute timeseries for {len(kvalues)} k values")

    # Create partial function with fixed parameters
    worker_func = partial(
        process_single_k_timeseries,
        s_init=s_init,
        t_grid_a=t_grid_a,
        t_grid_s=t_grid_s,
        num_time_points=num_time_points,
        endtime=endtime,
        swaptime=swaptime,
        recConformalTime=recConformalTime,
        H0=H0,
        OmegaLambda=OmegaLambda,
        OmegaM=OmegaM,
        OmegaR=OmegaR,
        OmegaK=OmegaK,
        deltaeta_max=deltaeta_max,
        k_deltaeta_target=k_deltaeta_target,
        atol=atol,
        rtol=rtol,
        num_variables=num_variables
    )

    # Use multiprocessing Pool to parallelize the computation
    with Pool(processes=n_processes) as pool:
        results = pool.map(worker_func, kvalues)

    print("\nParallel computation completed, assembling results...")

    # Unpack results
    all_ABC_solutions = []
    all_DEF_solutions = []
    all_GHI_solutions = []

    for ABC_sols_k, DEF_sols_k, GHI_sols_k in results:
        all_ABC_solutions.append(ABC_sols_k)
        all_DEF_solutions.append(DEF_sols_k)
        all_GHI_solutions.append(GHI_sols_k)

    # Convert lists of tensors to single large numpy arrays.
    # Full shapes:
    #   ABC: (num_k, num_variables, 6, num_times)
    #   DEF: (num_k, num_variables, 2, num_times)
    #   GHI: (num_k, num_variables,    num_times)
    all_ABC_solutions = np.array(all_ABC_solutions)
    all_DEF_solutions = np.array(all_DEF_solutions)
    all_GHI_solutions = np.array(all_GHI_solutions)
    print(f"Saving full arrays with num_variables={num_variables}")

    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("\nSaving all solution tensors to disk...")
    np.save(folder_path + 't_grid.npy', t_grid)
    np.save(folder_path + 'L70_kvalues.npy', kvalues)
    np.save(folder_path + 'L70_ABC_solutions.npy', all_ABC_solutions)
    np.save(folder_path + 'L70_DEF_solutions.npy', all_DEF_solutions)
    np.save(folder_path + 'L70_GHI_solutions.npy', all_GHI_solutions)
    print("All TimeSeries U matrices are calculated and saved successfully.")
