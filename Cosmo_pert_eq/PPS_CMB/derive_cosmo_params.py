params = [0.4444730086581741, -0.03940583926181416, 0.16336760287147203, 0.5615471381223595, 2.180497037062433, 0.9672453294986536, 0.07235242742234273] # Best-fit parameters, including As,ns,tau
params[2] = 0.16112337683135397  # best-fit found from fix3params

h = params[3]  # Hubble parameter
Omega_b = params[0] * params[2]  # Baryon density parameter
Omega_cdm = params[0] * (1 - params[2])  # Cold dark matter density parameter
omega_b = params[0] * params[2] * h**2  
omega_cdm = params[0] * (1 - params[2]) * h**2 
tau_reio = params[6]
A_s         = params[4] * 1e-09
n_s = params[5]
omega_k = params[1]  # This makes K>0 (closed universe)

print(f"Using cosmological parameters: OmegaM={params[0]}, OmegaK={params[1]}, omega_b_ratio={params[2]}, h={params[3]}, A_s={params[4]}, n_s={params[5]}, tau_reio={params[6]}")
print(f"Derived parameters: omega_b={omega_b}, omega_cdm={omega_cdm}, Omega_b={Omega_b}, Omega_cdm={Omega_cdm}, tau_reio={tau_reio}, A_s={A_s}, n_s={n_s}, omega_k={omega_k}")