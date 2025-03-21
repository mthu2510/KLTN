import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 

# Define constants
R = 20  # kpc
H = 4   # kpc
u0 = 7 * 1e5  # km/s to cm/s (1 km = 1e5 cm)
D0 = 1e28  # cm^2/s
r_val = 8  # kpc
z = 0  # Define z variable instead of substituting directly
E_vals = np.logspace(-1, 2, 100)  # Energy values from 0.1 GeV to 100 GeV
num_zeros = 100  # Number of first zeros of J0 to use

# Find the first zeros of J0(x)
zeros_j0 = sp.special.jn_zeros(0, num_zeros)

def func_gSNR(r):
    return 1.0 / (np.pi * R**2) # kpc^-2

r = np.linspace(0, R, 10000)
gr = func_gSNR(r)*r**0
dr = np.append(np.diff(r),0)

j0_n = sp.special.j0(zeros_j0[:,np.newaxis] * r[np.newaxis,:]/R)
coeff_gSNR = np.sum(r[np.newaxis,:] * gr[np.newaxis,:] * j0_n * dr[np.newaxis,:], axis=1)
coeff_gSNR *= (2.0 / (R * sp.special.j1(zeros_j0))**2)

gr_test = np.sum(j0_n * coeff_gSNR[:,np.newaxis], axis=0)

print(gr[0:10])
print(gr_test[0:10])

"""def compute_j_E(E_vals, num_terms=10):
    # Compute j(E) based on the zeros of the Bessel function of order 0
    D_E = D0 * (E_vals / 1.0) ** (1/3)  # D(E) according to the problem
    j_E = np.zeros_like(E_vals)

    for i, E in enumerate(E_vals):
        sum_term = 0
        for n in range(num_terms):
            zeta_n = zeros_J0[n]
            C_n = (zeta_n ** 2) * D_E[i] / R**2
            S_n = np.sqrt(u0**2 / (D_E[i]**2) + 4 * zeta_n**2 / R**2)

            A_n = - (g_SNR * D_E[i]) / (2 * np.exp(S_n * H / 2) * np.sinh(S_n * H / 2) * (u0 + D_E[i] * S_n / np.tanh(S_n * H / 2)))
            f_nz_0 = -2 * A_n * np.exp(u0 * z / (2 * D_E[i]) + S_n * H / 2) * np.sinh(S_n * (H - z) / 2)
            
            sum_term += np.abs(A_n) * np.exp(-C_n * z) * np.sin(zeta_n * r_val / R) * f_nz_0

        v = np.sqrt(2 * E * 1.6e-6 / 9.11e-28)  # Compute particle velocity (cm/s) from energy (1 GeV = 1.6e-6 erg)
        j_E[i] = v * sum_term / (4 * np.pi)
    
    return j_E

# Compute j(E)
j_E_vals = compute_j_E(E_vals)

# Plot the graph
plt.figure(figsize=(8, 6))
plt.loglog(E_vals, j_E_vals, label=r'$j(E)$ vs $E$')
plt.xlabel('E (GeV)')
plt.ylabel('j(E)')
plt.title('Particle Spectrum j(E)')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()"""