# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:21:56 2021

@author: MRose
"""


from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import root_scalar
import numpy as np
from math import *

#working in units 8piG = Lambda = c = hbar = kB = 1 throughout
def compute_X_recs(params, z_rec, folder_path):
    """
    Compute perturbation values at recombination for perturbation analysis.

    Parameters:
    -----------
    params : list or tuple
        [OmegaM, OmegaK, omega_b_ratio, h] cosmological parameters
    z_rec : float
        Recombination redshift
    folder_path : str
        Path to save output files
    nu_spacing : int, optional
        Spacing for allowed K values (default: 4)

    Returns:
    --------
    list : list containing recombination values for each k in kvalues
    """
    ###############################################################################
    # Unpack parameters
    OmegaM, OmegaK, omega_b_ratio, h = params
    Omega_gamma_h2 = 2.47e-5 # photon density 
    Neff = 3.046

    OmegaR = (1 + Neff * (7/8) * (4/11)**(4/3)) * Omega_gamma_h2 / h**2
    OmegaLambda = 1 - OmegaM - OmegaK - OmegaR
    ###############################################################################

    #set tolerances
    atol = 1e-13;
    rtol = 1e-13;
    stol = 1e-10;
    num_variables = 200; # number of pert variables
    swaptime = 2; #set time when we swap from s to sigma
    H0 = 1/np.sqrt(3*OmegaLambda); #we are working in units of Lambda=c=1
    Hinf = H0*np.sqrt(OmegaLambda);

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


    #``````````````````````````````````````````````````````````````````````````````
    #RECOMBINATION CONFORMAL TIME
    #```````````````````````````````````````````````````````````````````````````````

    # Build continuous a -> t interpolant (a is monotonically increasing)
    _interp_t_from_a = interp1d(sol_a.y[0], sol_a.t, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')

    a_rec = 1. / (1 + z_rec)  # scale factor at recombination
    recConformalTime = float(_interp_t_from_a(a_rec))
    print(f"Recombination conformal time: {recConformalTime}")


    #-------------------------------------------------------------------------------
    # Define derivative functions
    #-------------------------------------------------------------------------------
    def dX1_dt(t, X):
        adot = X[0]**2*H0*np.sqrt((OmegaLambda + OmegaK/abs(X[0]**2) + OmegaM/abs(X[0]**3) + OmegaR/abs(X[0]**4)))

        rho_m = 3*(H0**2)*OmegaM/(abs(X[0]**3))
        rho_r = 3*(H0**2)*OmegaR/(abs(X[0]**4))

        phidot = - (adot/X[0])*X[1] - ((4/3)*rho_r*X[4] + rho_m*X[5])*(X[0]**2/2)
        drdot = (4/3)*(3*phidot + (k**2)*X[4]);
        dmdot = 3*phidot + X[5]*(k**2);
        vrdot = -(X[1] + X[2]/4);
        vmdot = - (adot/X[0])*X[5] - X[1];

        return [adot, phidot, drdot, dmdot, vrdot, vmdot]

    #------------------------------------------------------------------------------------
    # Now find and store rec values
    #------------------------------------------------------------------------------------

    kvalues = np.load(folder_path+'L70_kvalues.npy');
    recValuesList = [];

    for i in range(len(kvalues)):
        
        k = kvalues[i];
        print(k)
        
        #------------------------------------------------------------------------------
        # Set up actual recombination values
        #------------------------------------------------------------------------------

        phi1 = -(H0*OmegaM)/(16*(OmegaR**0.5));
        phi2 = (1/60)*(-2*k**2 - 8*OmegaK/OmegaLambda + (9*OmegaM**2)/(16*OmegaLambda*OmegaR));
        phi3 = (4*OmegaM*OmegaR*(71*k**2*OmegaLambda + 174*OmegaK) - 63*OmegaM**3)/(17280*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
        phi4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1546*k**2*OmegaLambda) + 96*OmegaR**2*(80*OmegaK**2 + 32*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-40*OmegaR + k**4*OmegaLambda)))/(725760*OmegaR**2*OmegaLambda**2)

        dr1 = -(H0*OmegaM)/(4*(OmegaR**0.5));
        dr2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(240*OmegaR*OmegaLambda);
        dr3 = (OmegaM*OmegaR*(696*OmegaK + 404*k**2*OmegaLambda) - 63*OmegaM**3)/(4320*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2));
        dr4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1924*k**2*OmegaLambda) + 48*OmegaR**2*(160*OmegaK**2 + 176*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-80*OmegaR + 23*k**4*OmegaLambda)))/(181440*OmegaR**2*OmegaLambda**2);
        
        dm1 = - (3*H0*OmegaM)/(16*(OmegaR**0.5));
        dm2 = (9*OmegaM**2 - 112*OmegaR*OmegaLambda*k**2 - 128*OmegaR*OmegaK)/(320*OmegaR*OmegaLambda);
        dm3 = (OmegaM*OmegaR*(404*k**2*OmegaLambda + 696*OmegaK) - 63*OmegaM**3)/(5760*np.sqrt(3)*OmegaLambda**(3/2)*OmegaR**(3/2));
        dm4 = (315*OmegaM**4 - OmegaM**2*OmegaR*(3915*OmegaK + 1924*k**2*OmegaLambda) + 24*OmegaR**2*(320*OmegaK**2 - 480*OmegaR*OmegaLambda + 247*k**2*OmegaK*OmegaLambda + 33*k**4*OmegaLambda**2))/(241920*OmegaR**2*OmegaLambda**2);
        
        vr1 = -1/2;
        vr2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
        vr3 = (-OmegaM**2 + 8*OmegaR*OmegaLambda*k**2)/(160*OmegaR*OmegaLambda) + 4.*OmegaK/(45*OmegaLambda);
        vr4 = (63*OmegaM**3 - 8*OmegaM*OmegaR*(87*OmegaK + 43*k**2*OmegaLambda))/(34560*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2));
        vr5 = (-63*OmegaM**4 + OmegaM**2*OmegaR*(783*OmegaK + 347*k**2*OmegaLambda) - 24*OmegaR**2*(64*OmegaK**2 + 48*k**2*OmegaK*OmegaLambda + 3*OmegaLambda*(-32*OmegaR + 5*k**4*OmegaLambda)))/(362880*OmegaR**2*OmegaLambda**2);

        vm1 = -1/2;
        vm2 = OmegaM/(16*np.sqrt(3*OmegaR*OmegaLambda));
        vm3 = (-3*OmegaM**2 + 4*OmegaR*OmegaLambda*k**2)/(480*OmegaR*OmegaLambda) + 17*OmegaK/(360*OmegaLambda);
        vm4 = (63*OmegaM**3 - 32*OmegaM*OmegaR*(15*OmegaK + 4*k**2*OmegaLambda))/(34560*np.sqrt(3)*OmegaR**(3/2)*OmegaLambda**(3/2));
        vm5 = (-63*OmegaM**4 + 2*OmegaM**2*OmegaR*(297*OmegaK + 79*k**2*OmegaLambda) - 24*OmegaR**2*(43*OmegaK**2 + 13*k**2*OmegaK*OmegaLambda + OmegaLambda*(-96*OmegaR + k**4*OmegaLambda)))/(362880*OmegaR**2*OmegaLambda**2);

        #set initial conditions
        phi0_guess = 1 + phi1*t0 + phi2*t0**2 + phi3*t0**3 + phi4*t0**4; #t0 from above in "background equations section"
        dr0 = -2 + dr1*t0 + dr2*t0**2 + dr3*t0**3 + dr4*t0**4;
        dm0 = -1.5 + dm1*t0 + dm2*t0**2 + dm3*t0**3 + dm4*t0**4;
        vr0 = vr1*t0 + vr2*t0**2 + vr3*t0**3 + vr4*t0**4 + vr5*t0**5;
        vm0 = vm1*t0 + vm2*t0**2 + vm3*t0**3 + vm4*t0**4 + vm5*t0**5;

        #------------------------------------------------------------------------------
        # Define helper function: solve ODE for given phi0 and return values at recombination
        #------------------------------------------------------------------------------
        def solve_for_phi0(phi0_input):
            """Solve ODE with given phi0 and return solution at recombination."""
            X0 = [a_Bang, phi0_input, dr0, dm0, vr0, vm0];
            sol3 = solve_ivp(dX1_dt, [t0, recConformalTime], X0, method='LSODA', atol=atol, rtol=rtol);
            return sol3

        #------------------------------------------------------------------------------
        # Define residual function for root-finding
        #------------------------------------------------------------------------------
        def residual(phi0_input):
            """
            Compute residual: phi_rec - phi_constraint
            The constraint should be satisfied at recombination.
            """
            # Solve ODE with this phi0
            sol3 = solve_for_phi0(phi0_input)

            # Extract values at recombination (last point)
            a_rec = sol3.y[0, -1]
            phi_rec = sol3.y[1, -1]
            dr_rec = sol3.y[2, -1]
            dm_rec = sol3.y[3, -1]
            vr_rec = sol3.y[4, -1]
            vm_rec = sol3.y[5, -1]
            t_rec = sol3.t[-1]

            # Compute the constraint value at recombination
            adot_rec = da_dt(t_rec, a_rec)
            phi_constraint = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
                (-3*adot_rec/a_rec*vm_rec + dm_rec)*OmegaM/a_rec +
                (-4*adot_rec/a_rec*vr_rec + dr_rec)*OmegaR/a_rec**2
            )

            # Return residual
            return phi_rec - phi_constraint

        #------------------------------------------------------------------------------
        # Use root-finding to find the correct phi0
        #------------------------------------------------------------------------------
        print(f"Starting root-finding for phi0...")
        print(f"Initial guess: phi0_guess = {phi0_guess}")

        # Use root_scalar to find phi0 that satisfies the constraint
        # We need to provide a bracket or use a method that doesn't require one
        try:
            # Try bracketing method first
            result = root_scalar(residual, bracket=[phi0_guess*0.5, phi0_guess*1.5], method='brentq', xtol=1e-10)
            phi0_optimal = result.root
            print(f"Root-finding converged!")
            print(f"Optimal phi0 = {phi0_optimal}")
            print(f"Residual = {result.function_calls} function calls")
        except ValueError:
            # If bracket doesn't work, try secant method
            print("Bracket method failed, trying secant method...")
            result = root_scalar(residual, x0=phi0_guess, x1=phi0_guess*1.1, method='secant', xtol=1e-10)
            phi0_optimal = result.root
            print(f"Root-finding converged!")
            print(f"Optimal phi0 = {phi0_optimal}")

        #------------------------------------------------------------------------------
        # Solve with optimal phi0 and extract recombination values
        #------------------------------------------------------------------------------
        sol3_final = solve_for_phi0(phi0_optimal)

        # Extract values at recombination
        a_rec_final = sol3_final.y[0, -1]
        phi_rec_final = sol3_final.y[1, -1]
        dr_rec_final = sol3_final.y[2, -1]
        dm_rec_final = sol3_final.y[3, -1]
        vr_rec_final = sol3_final.y[4, -1]
        vm_rec_final = sol3_final.y[5, -1]

        # Verify constraint is satisfied
        adot_rec_final = da_dt(recConformalTime, a_rec_final)
        phi_constraint_final = - 3*H0**2 / (2*(k**2 + 3*OmegaK*H0**2)) * (
            (-3*adot_rec_final/a_rec_final*vm_rec_final + dm_rec_final)*OmegaM/a_rec_final +
            (-4*adot_rec_final/a_rec_final*vr_rec_final + dr_rec_final)*OmegaR/a_rec_final**2
        )

        print(f"\nVerification at recombination:")
        print(f"phi_rec = {phi_rec_final}")
        print(f"phi_constraint = {phi_constraint_final}")
        print(f"Difference = {abs(phi_rec_final - phi_constraint_final)}")

        # Store recombination values (format same as before)
        X00 = [a_rec_final, phi_rec_final, phi_rec_final, dr_rec_final, dm_rec_final,
            vr_rec_final, vm_rec_final];

        recValues = X00[1:num_variables+1];
        recValuesList.append(recValues);

    np.save(folder_path+'L70_recValues', recValuesList)