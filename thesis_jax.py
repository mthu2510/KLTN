import numpy as np
import jax.numpy as jnp
import jax.random as jr 
import matplotlib.pyplot as plt
from jax import jit
import scipy as sp 

# Define constants
pars = jnp.array([20, 4, 700.0e5, 1.0e12, 1.0e28, 8.0, 0]) 
# H, R (kpc), u0 (km/s to cm/s), Q0, D0, r_vals, z
E_vals = jnp.logspace(9.0, 11.0, 100) # 1-100 GeV (10^9-10^11 eV)
E_fixed = 1.0e10 # 10 GeV
num_zeros = 500  # Number of first zeros of J0 to use

# Find the first zeros of J0(x)
zeros_j0 = sp.special.jn_zeros(0, num_zeros) # zeta_n

# Surface density of SNRs for the honomogeneous case
def func_gSNR(r): # r (kpc)
    gSNR = jnp.where(
        r <= 15.0,
        r**0 / (jnp.pi * 15.0**2),
        0.0
    )
    return gSNR # kpc^-2

def func_gSNR_smooth(r, Rs = 15.0, eps = 0.1):  # Rs: cutoff radius, eps: smoothing width
    return (1.0 / (jnp.pi * Rs**2)) * 0.5 * (1.0 - jnp.tanh((r - Rs) / eps))

# Surface density of SNRs from Yusifov et al. 2004
def func_gSNR_YUK04(r):
# r (pc)
    r = jnp.array(r) * 1.0e-3 # kpc
    gSNR = jnp.where(
        r < 15.0,
        jnp.power((r + 0.55) / 9.05, 1.64) * jnp.exp(-4.01 * (r - 8.5) / 9.05) / 5.95828e+8,
        0.0
    )    
    return gSNR # pc^-2

def func_gSNR_YUK04_smooth(r, Rs = 15.0, eps = 0.1):  # Rs: cutoff radius, eps: smoothing width
    g = jnp.power((r + 0.55) / 9.05, 1.64) * jnp.exp(-4.01 * (r - 8.5) / 9.05) / 5.95828e+8
    g_smooth = g * 0.5 * (1.0 - jnp.tanh((r - Rs) / eps))
    return g_smooth * 1e6 # kpc^-2

# Verify the accuracy of the function g_SNR
def func_coeff_gSNR(pars, zeros_j0):
    r = jnp.linspace(0, pars[1], 10000)
    gr = func_gSNR_YUK04_smooth(r) * r**0
    dr = jnp.append(jnp.diff(r),0)

    j0_n = jnp.special.jv(0, zeros_j0[:,jnp.newaxis] * r[jnp.newaxis,:] / pars[1])

    coeff_gSNR = jnp.sum(r[jnp.newaxis,:] * gr[jnp.newaxis,:] * j0_n * dr[jnp.newaxis,:], axis=1)
    coeff_gSNR *= (2.0 / (pars[1] * jnp.special.jv(1, zeros_j0))**2)

    gr_test = jnp.sum(j0_n * coeff_gSNR[:,jnp.newaxis], axis=0) 

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
    plt.savefig('fg_gSNR_jax.png')
    plt.close()

    return coeff_gSNR

def Q_E_func(pars, E):
    return pars[3] * (E / 1.0e9) ** (-2.4) # eV

def D_E_func(pars, E):
    return pars[4] * (E / 1.0e9) ** (1/3)  # cm^2/s

def relativistic_velocity(E):  # E in eV
    erg = E * 1.6e-12
    m = 1.67e-24  # g, proton mass
    c = 3e10      # cm/s
    return c * jnp.sqrt(1 - (m * c**2 / (erg + m * c**2))**2)

@jit
def compute_j_E(pars, zeros_j0, E_vals, g_SNR):
    Q_E = Q_E_func(pars, E_vals) 
    D_E = D_E_func(pars, E_vals)

    S_n = jnp.sqrt(pars[2]**2 / (D_E[:,jnp.newaxis]**2) + 4 * zeros_j0[jnp.newaxis,:]**2 / pars[1]**2)
    coth_SnH = 1.0 / jnp.tanh(S_n * pars[0] / 2.0)

    J0_rval = jnp.special.jv(0, zeros_j0 * pars[5] / pars[1])  # Compute J0 at r_val
    
    f_z = jnp.sum((g_SNR[jnp.newaxis,:] * J0_rval[jnp.newaxis,:] * jnp.exp(pars[2] * pars[6] / (2.0 * D_E[:,jnp.newaxis])) * jnp.sinh(S_n * (pars[0] - pars[6]) / 2.0)) \
                                / (jnp.sinh(S_n * pars[0] / 2.0) * (pars[2] + D_E[:,jnp.newaxis] * S_n * coth_SnH)), axis = 1)
    
    f_z *= Q_E

    vA = relativistic_velocity(E_vals)
    j_E = vA * f_z / (4 * jnp.pi)

    return j_E 

def plot_jE (E_vals, j_E_vals): 
    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.loglog(E_vals, j_E_vals, label = f'$j(E)$ with vA = 7 km/s')

    # Plot from the data:
    filename = 'plot_data_flux_p_AMS.dat'
    Ea, jE_AMS = np.loadtxt(filename, unpack=True, usecols=[0,1])

    plt.plot(Ea, jE_AMS, 'ko', label = f'$j(E)$ from the data')

    plt.xlabel('E [eV]')
    plt.ylabel('j(E) [eV^{-1} cm^{-2} s^{-1}]')
    plt.title('Particle Spectrum j(E)')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth = 0.5)
    plt.savefig('fg_j(E)_jax.png')
    plt.close()
    return Ea, jE_AMS

# Compute and plot j(E)
g_SNR = func_coeff_gSNR(pars, zeros_j0) # shape (N,)
j_E_vals = compute_j_E(pars, zeros_j0, E_vals, g_SNR)
Ea, jE_AMS = plot_jE (E_vals, j_E_vals)

def index_j(E_vals, j_E_vals):
    return - np.log10(j_E_vals[0] / j_E_vals[-1]) / np.log10(E_vals[0] / E_vals[-1])

# Compute the index of the slope 
alpha = index_j(E_vals, j_E_vals)
print(f"Spectral index α ≈ {alpha:.3f}")

alpha_data = index_j(Ea, jE_AMS)
print(f"AMS-02 Spectral index α_data ≈ {alpha_data:.3f}")

def compute_jE_fixed_E (pars, E_fixed, g_SNR):
    r = jnp.linspace(0, pars[1], 200) # (200,)
    z = jnp.linspace(0, pars[0], 200) # (200,)

    # Create meshgrid of r and z
    R_grid, Z_grid = jnp.meshgrid(r, z, indexing='ij')  # (200, 200)

    Q_E = Q_E_func(pars, E_fixed)
    D_E = D_E_func(pars, E_fixed)

    S_n = jnp.sqrt(pars[2]**2 / (D_E**2) + 4 * zeros_j0**2 / pars[1]**2) # shape (N,)
    coth_SnH = 1.0 / jnp.tanh(S_n * pars[2] / 2.0) # shape (N,)

    J0 = jnp.special.jv(0, zeros_j0[jnp.newaxis, jnp.newaxis, :] * R_grid[:, :, jnp.newaxis] / pars[1]) 
    
    f_z = jnp.sum((g_SNR[jnp.newaxis, jnp.newaxis, :] * J0 * jnp.exp(pars[2] * Z_grid[:, :, jnp.newaxis] / (2.0 * D_E)) * jnp.sinh(S_n[jnp.newaxis, jnp.newaxis, :] * (pars[0] - Z_grid[:, :, jnp.newaxis]) / 2.0)) \
                                / (jnp.sinh(S_n[jnp.newaxis, jnp.newaxis, :] * pars[0] / 2.0) * (pars[2] + D_E * S_n[jnp.newaxis, jnp.newaxis, :] * coth_SnH[jnp.newaxis, jnp.newaxis, :])), axis = 2)
    
    f_z *= Q_E
    vA = relativistic_velocity(E_fixed)
    j_E = vA * f_z / (4 * jnp.pi)

    return r, z, R_grid, Z_grid, j_E

def plot_2D_fixed_E(r, z, R_grid, Z_grid, j_E_vals):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [2, 1]})

    contour = axs[0, 0].contourf(R_grid, Z_grid, j_E_vals, levels=50, cmap = 'viridis')
    plt.colorbar(contour, ax=axs[0, 0])
    axs[0, 0].set_xlabel('r [kpc]')
    axs[0, 0].set_ylabel('z [kpc]')
    axs[0, 0].set_title('2D map of j(E) at E = 10 GeV')
    axs[0, 0].grid(True)

    axs[1, 0].plot(r, j_E_vals[:,0], color='red')
    axs[1, 0].set_xlabel('r [kpc]')
    axs[1, 0].set_ylabel('j(E) [eV$^{-1}$ cm^{-2} s^{-1}]')
    axs[1, 0].set_title('2D map of j(E) at E = 10 GeV and z = 0')
    axs[1, 0].grid(True)

    axs[1, 1].plot(z, j_E_vals[0,:], color='red')
    axs[1, 1].set_xlabel('z [kpc]')
    axs[1, 1].set_ylabel('j(E) [eV^{-1} cm^{-2} s^{-1}]')
    axs[1, 1].set_title('2D map of j(E) at E = 10 GeV and r = 0')
    axs[1, 1].grid(True)

    # Loại bỏ vị trí không cần thiết (hàng 1, cột 2)
    fig.delaxes(axs[0, 1])

    plt.tight_layout()
    plt.savefig('fg_jE_fixed_E_jax.png')
    plt.close()
    
r, z, R_grid, Z_grid, j_E_vals = compute_jE_fixed_E (pars, E_fixed, g_SNR)
plot_2D_fixed_E(r, z, R_grid, Z_grid, j_E_vals) 