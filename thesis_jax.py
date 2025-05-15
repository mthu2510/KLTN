import numpy as np
"""from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)"""
import os
os.environ['JAX_ENABLE_X64'] = 'True'
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp
from jax import jit

# Define constants
pars = jnp.array([
    4.0,                  # H (kpc)
    20.0,                 # R (kpc)
    7.0 * 1e5 / 3.086e21, # u0 (cm/s -> kpc/s)
    1.0e12,               # Q0 (eV^{-1} s^{-1} kpc^{-2})
    1.0e28,               # D0 (cm^2/s)
    8.0                   # r_vals (kpc)
])
mp = 9.38272e8  # proton rest mass energy in eV
z = 0
E_vals = jnp.logspace(9.0, 11.0, 100) # 1-100 GeV (10^9-10^11 eV)
E_fixed = 1.0e10 # 10 GeV
num_zeros = 100  # Number of first zeros of J0 to use

# Find the first zeros of J0(x)
zeros_j0 = sp.special.jn_zeros(0, num_zeros) # zeta_n

# Zeroth order Bessel function of first kind
@jit
def j0(x):
    def small_x(x):
        z = x * x
        num = 57568490574.0 + z * (-13362590354.0 + z * (651619640.7 +
              z * (-11214424.18 + z * (77392.33017 + z * (-184.9052456)))))
        den = 57568490411.0 + z * (1029532985.0 + z * (9494680.718 +
              z * (59272.64853 + z * (267.8532712 + z * 1.0))))
        return num / den

    def large_x(x):
        y = 8.0 / x
        y2 = y * y
        ans1 = 1.0 + y2 * (-0.1098628627e-2 + y2 * (0.2734510407e-4 +
               y2 * (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
        ans2 = -0.1562499995e-1 + y2 * (0.1430488765e-3 +
               y2 * (-0.6911147651e-5 + y2 * (0.7621095161e-6 -
               y2 * 0.934935152e-7)))
        return jnp.sqrt(0.636619772 / x) * (jnp.cos(x - 0.785398164) * ans1 - y * jnp.sin(x - 0.785398164) * ans2)

    return jnp.where(x < 5.0, small_x(x), large_x(x))

# First order Bessel function of first kind
@jit
def j1(x):
    def small_x(x):
        z = x * x
        num = x * (72362614232.0 + z * (-7895059235.0 + z * (242396853.1 +
              z * (-2972611.439 + z * (15704.48260 + z * (-30.16036606))))))
        den = 144725228442.0 + z * (2300535178.0 + z * (18583304.74 +
              z * (99447.43394 + z * (376.9991397 + z * 1.0))))
        return num / den

    def large_x(x):
        y = 8.0 / x
        y2 = y * y
        ans1 = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 +
               y2 * (0.2457520174e-5 - y2 * 0.240337019e-6)))
        ans2 = 0.04687499995 + y2 * (-0.2002690873e-3 +
               y2 * (0.8449199096e-5 + y2 * (-0.88228987e-6 +
               y2 * 0.105787412e-6)))
        return jnp.sqrt(0.636619772 / x) * (jnp.cos(x - 2.356194491) * ans1 - y * jnp.sin(x - 2.356194491) * ans2)

    return jnp.where(x < 5.0, small_x(x), large_x(x))


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
    r = jnp.array(r) #* 1.0e-3 # kpc
    gSNR = jnp.where(
        r < 15.0,
        jnp.power((r + 0.55) / 9.05, 1.64) * jnp.exp(-4.01 * (r - 8.5) / 9.05) / 5.95828e+8,
        0.0
    )    
    return gSNR * 1.0e6 # kpc^-2

def func_gSNR_YUK04_smooth(r, Rs = 15.0, eps = 0.1):  # Rs: cutoff radius, eps: smoothing width
    g = jnp.power((r + 0.55) / 9.05, 1.64) * jnp.exp(-4.01 * (r - 8.5) / 9.05) / 5.95828e+8
    g_smooth = g * 0.5 * (1.0 - jnp.tanh((r - Rs) / eps))
    return g_smooth * 1.0e6 # kpc^-2

# Verify the accuracy of the function g_SNR
def func_coeff_gSNR(pars, zeros_j0):
    r = jnp.linspace(0, pars[1], 10000)     # kpc
    gr = func_gSNR_YUK04_smooth(r) * r**0   # kpc^-2
    dr = jnp.append(jnp.diff(r),0)

    j0_n = j0(zeros_j0[:,jnp.newaxis] * r[jnp.newaxis,:] / pars[1])

    coeff_gSNR = jnp.sum(r[jnp.newaxis,:] * gr[jnp.newaxis,:] * j0_n * dr[jnp.newaxis,:], axis=1)
    coeff_gSNR *= (2.0 / (pars[1] * j1(zeros_j0))**2)

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
    return coeff_gSNR # kpc^-2

"""def Q_E_func(pars, E):
    return pars[3] * (E / 1.0e9) ** (-2.4) # eV^−1 s^−1 kpc^−2"""

def Q_E_func(pars, E): 
    p=jnp.sqrt((E+mp)**2-mp**2) # eV
    vp=p/(E+mp)
    xiSNR = 0.1 
    alpha = 4.4

    # Injection spectrum of sources
    xmin=jnp.sqrt((1.0e8+mp)**2-mp**2)/mp
    xmax=jnp.sqrt((1.0e14+mp)**2-mp**2)/mp
    x=jnp.logspace(jnp.log10(xmin),jnp.log10(xmax),5000)
    Gam=jnp.trapezoid(x**(2.0-alpha)*(jnp.sqrt(x**2+1.0)-1.0),x)

    RSNR=0.03 # yr^-1 -> SNR rate
    ENSR=1.0e51*6.242e+11 # eV -> Average kinetic energy of SNRs
    QE=(xiSNR*RSNR*ENSR/(mp**2*vp*Gam))*(p/mp)**(2.0-alpha) / (365.0*86400.0) 

    return QE # eV^-1 s^-1

def D_E_func(pars, E):
    D_E = pars[4] * (E / 1.0e9) ** (1/3)  # cm^2/s
    return D_E / (3.086e21)**2 # kpc^2/s

def relativistic_velocity(E):  # E in eV
    c = 3e10          # cm/s
    return c * jnp.sqrt(1 - 1 / ((E + mp) / mp) **2) # cm/s

@jit
def compute_j_E(pars, zeros_j0, E_vals, g_SNR):
    Q_E = Q_E_func(pars, E_vals) 
    D_E = D_E_func(pars, E_vals)
    
    # pars[2] = u0 = cm/s
    # D_E = cm^2/s 
    # (pars[2]/D_E)^2 = ((cm/s)/(cm^2/s))^2 = (1/cm)^2
    # zeros_j0 = 1
    # pars[1] = R = kpc
    # pars[0] = H = kpc

    S_n = jnp.sqrt(((pars[2] / D_E[:,jnp.newaxis]) **2) + 4 * (zeros_j0[jnp.newaxis,:] / pars[1]) **2) # 1/kpc
    coth_SnH = 1.0 / jnp.tanh(S_n * pars[0] / 2.0) # 1/kpc * kpc = 1

    J0_rval = j0(zeros_j0 * pars[5] / pars[1])  # Compute J0 at r_val
    
    f_z = jnp.sum((g_SNR[jnp.newaxis,:] * J0_rval[jnp.newaxis,:] * jnp.exp(pars[2] * z / (2.0 * D_E[:,jnp.newaxis])) * jnp.exp(-S_n * z / 2) - jnp.exp(-S_n * pars[0] + S_n * z / 2)) \
                                / ((1 - jnp.exp(-S_n * pars[0])) * (pars[2] + D_E[:,jnp.newaxis] * S_n * coth_SnH)), axis = 1)
    
    f_z *= Q_E 

    vp = relativistic_velocity(E_vals)
    j_E = vp * f_z / (4 * jnp.pi) 

    return j_E * 1.0e-3 / (3.086e21)**3 # eV^-1 s-1 cm^-2 sr^-1
    #return S_n * (pars[0] - z) / 2.0

def plot_jE (E_vals, j_E_vals): 
    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.loglog(E_vals, jnp.abs(j_E_vals), label = f'$j(E)$ with u0 = 7 km/s')

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
print(g_SNR)
j_E_vals = compute_j_E(pars, zeros_j0, E_vals, g_SNR)
print(j_E_vals)
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

    J0 = j0(zeros_j0[jnp.newaxis, jnp.newaxis, :] * R_grid[:, :, jnp.newaxis] / pars[1]) 
    
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