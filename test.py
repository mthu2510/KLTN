import jax.numpy as jnp
from jax import jit

@jit
def diffusion(u, dx, dt, D):
    """
    Mô phỏng phương trình khuếch tán 1D với bước tiến thời gian dt.
    
    Args:
        u: Mảng (array) giá trị u(x, t) tại thời điểm hiện tại.
        dx: Bước lưới theo không gian (x).
        dt: Bước lưới theo thời gian (t).
        D: Hệ số khuếch tán.
        
    Returns:
        Mảng giá trị u(x, t+dt) sau một bước thời gian.
    """
    # Tính gradient bậc hai (d^2u/dx^2) sử dụng sai phân hữu hạn
    d2u_dx2 = (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / dx**2
    
    # Cập nhật giá trị u theo phương trình khuếch tán
    return u + dt * D * d2u_dx2

# Lưới không gian và thời gian
nx = 100  # Số điểm không gian
dx = 0.1  # Khoảng cách giữa các điểm
dt = 0.001  # Bước thời gian
nt = 500  # Số bước thời gian
D = 0.1  # Hệ số khuếch tán

# Điều kiện ban đầu: Gauss hoặc Dirac delta
x = jnp.linspace(0, nx*dx, nx)
u0 = jnp.exp(-((x - 5)**2) / 0.5)  # Hàm Gauss tại x = 5
# Khởi tạo u ban đầu
u = u0.copy()

# Lưu trữ kết quả tại mỗi bước thời gian
results = [u]

for t in range(nt):
    u = diffusion(u, dx, dt, D)  # Cập nhật giá trị u
    results.append(u)
import matplotlib.pyplot as plt

for i in range(0, nt, 50):  # Vẽ mỗi 50 bước thời gian
    plt.plot(x, results[i], label=f't={i*dt:.2f}')

plt.title("Quá trình khuếch tán theo thời gian")
plt.xlabel("Vị trí x")
plt.ylabel("u(x, t)")
plt.legend()
plt.show()
