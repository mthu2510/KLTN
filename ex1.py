import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros

# Định nghĩa các hằng số
R = 20  # kpc
H = 4   # kpc
u0 = 7 * 1e5  # km/s chuyển thành cm/s (1 km = 1e5 cm)
D0 = 1e28  # cm^2/s
g_SNR = 2e-9  # pc^-2
r_val = 8  # kpc
z = 0  # Đặt biến z thay vì thế trực tiếp
E_vals = np.logspace(-1, 2, 100)  # Giá trị năng lượng từ 0.1 GeV đến 100 GeV
num_zeros = 10  # Số nghiệm đầu tiên của J0 cần sử dụng

# Tìm các nghiệm đầu tiên của J0(x)
zeros_J0 = jn_zeros(0, num_zeros)

def compute_j_E(E_vals, num_terms=10):
    """Tính giá trị j(E) dựa trên các nghiệm của hàm Bessel bậc 0"""
    D_E = D0 * (E_vals / 1.0) ** (1/3)  # D(E) theo bài toán
    j_E = np.zeros_like(E_vals)

    for i, E in enumerate(E_vals):
        sum_term = 0
        for n in range(num_terms):
            zeta_n = zeros_J0[n]
            C_n = (zeta_n ** 2) * D_E[i]**2 / R**2
            S_n = np.sqrt(u0**2 / (D_E[i]**2) + 4 * zeta_n**2 / R**2)

            A_n = - (g_SNR * D_E[i]) / (2 * np.exp(S_n * H / 2) * np.sinh(S_n * H / 2) * (u0 + D_E[i] * S_n / np.tanh(S_n * H / 2)))
            f_nz_0 = -2 * A_n * np.exp(u0 * z / (2 * D_E[i]) + S_n * H / 2) * np.sinh(S_n * (H - z) / 2)
            
            sum_term += np.abs(A_n) * np.exp(-C_n * z) * np.sin(zeta_n * r_val / R) * f_nz_0

        v = np.sqrt(2 * E * 1.6e-6 / 9.11e-28)  # Tính vận tốc hạt (cm/s) từ năng lượng (1 GeV = 1.6e-6 erg)
        j_E[i] = v * sum_term / (4 * np.pi)
    
    return j_E

# Tính giá trị j(E)
j_E_vals = compute_j_E(E_vals)

# Vẽ đồ thị
plt.figure(figsize=(8, 6))
plt.loglog(E_vals, j_E_vals, label=r'$j(E)$ vs $E$')
plt.xlabel('E (GeV)')
plt.ylabel('j(E)')
plt.title('Phổ hạt j(E)')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()