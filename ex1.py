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
E_vals = np.logspace(9.0, 11.0, 100) # 1-100 GeV (10^9-10^11 eV)
num_zeros = 1000  # Number of first zeros of J0 to use

# Find the first zeros of J0(x)
zeros_j0 = sp.special.jn_zeros(0, num_zeros) # zeta_n

# Surface density of SNRs for the honomogeneous case
def func_gSNR(r): # r (kpc)
    gSNR = np.where(
        r<=15.0,
        r**0/(np.pi*15.0**2),
        0.0
    )
    return gSNR # kpc^-2

def func_gSNR_s(r, Rs = 15.0, eps = 0.3):  # Rs: cutoff radius, eps: smoothing width
    return (1.0 / (np.pi * Rs**2)) * 0.5 * (1.0 - np.tanh((r - Rs) / eps))

# Verify the accuracy of the function g_SNR
def func_coeff_gSNR(R, zeros_j0):
    r = np.linspace(0, R, 10000)
    gr = func_gSNR_s(r) * r**0
    dr = np.append(np.diff(r),0)

    j0_n = sp.special.j0(zeros_j0[:,np.newaxis] * r[np.newaxis,:] / R)

    coeff_gSNR = np.sum(r[np.newaxis,:] * gr[np.newaxis,:] * j0_n * dr[np.newaxis,:], axis=1)
    coeff_gSNR *= (2.0 / (R * sp.special.j1(zeros_j0))**2)

    gr_test = np.sum(j0_n * coeff_gSNR[:,np.newaxis], axis=0) 

    # Plot gr vs gr_test for validation
    plt.figure(figsize=(8, 6))
    plt.plot(r, gr, label='gr')
    plt.plot(r, gr_test, label='gr_test', linestyle='dashed')
    # plt.ylim()
    plt.xlabel('r (kpc)')
    plt.ylabel('g_SNR')
    plt.title('Test g_SNR')
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig('fg_gSNR.png')
    plt.close()

    return coeff_gSNR

def Q_E_func(Q0, E):
    return Q0 * (E / 1.0e9) ** (-2.4) # eV

Q_E = Q_E_func(1.0e16, E_vals)

def D_E_func(D0, E):
    return D0 * (E / 1.0e9) ** (1/3)  # cm^2/s

D_E = D_E_func(D0, E_vals)

def compute_j_E(R, zeros_j0, u0, D_E, H, r_val, z, Q_E):
    # Compute j(E) based on the zeros of the Bessel function of order 0
    
    g_SNR = func_coeff_gSNR(R, zeros_j0)

    S_n = np.sqrt(u0**2 / (D_E[:,np.newaxis]**2) + 4 * zeros_j0[np.newaxis,:]**2 / R**2)
    coth_SnH = 1.0 / np.tanh(S_n * H / 2.0)

    J0_rval = sp.special.j0(zeros_j0 * r_val / R)  # Compute J0 at r_val
    
    f_z = np.sum((g_SNR[np.newaxis,:] * J0_rval[np.newaxis,:] * np.exp(u0 * z / (2.0 * D_E[:,np.newaxis])) * np.sinh(S_n * (H - z) / 2.0)) \
                                / (np.sinh(S_n * H / 2.0) * (u0 + D_E[:,np.newaxis] * S_n * coth_SnH)), axis = 1)
    
    f_z *= Q_E

    # v = np.sqrt(2 * E_vals * 1.6e-6 / 9.11e-28)  # Compute particle velocity (cm/s) from energy (1 GeV = 1.6e-6 erg)
    vA = u0 
    j_E = vA * f_z / (4 * np.pi)

    return j_E 

def plot_jE (E_vals, j_E_vals): 
    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.loglog(E_vals, j_E_vals, label=r'$j(E)$ vs $E$ from the model')

    # Plot from the data:
    filename = 'plot_data_flux_p_AMS.dat'
    Ea, jE_AMS = np.loadtxt(filename, unpack=True, usecols=[0,1])

    plt.plot(Ea, jE_AMS, label=r'$j(E)$ vs $E$ from the data')

    plt.xlabel('E (GeV)')
    plt.ylabel('j(E)')
    plt.title('Particle Spectrum j(E)')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth = 0.5)
    plt.savefig('fg_j(E).png')
    plt.close()


def index_j(E_vals, j_E_vals):
    return - np.log10(j_E_vals[0] / j_E_vals[-1]) / np.log10(E_vals[0] / E_vals[-1])

# Compute j(E)
j_E_vals = compute_j_E(E_vals, num_zeros)
plot_jE (E_vals, j_E_vals)

# Compute the index of the slope 
alpha = index_j(E_vals, j_E_vals)
print(f"Spectral index α ≈ {alpha:.3f}")
