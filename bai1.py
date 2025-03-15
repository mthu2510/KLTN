import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import*
#mpl.rc("text",usetex=True)

fs = 22

def func_Q (Q0, E):

    Q = Q0 * (E/1.0e9) ** (-2.4)
    return Q

def func_D (E):

    D = 1.0e28 * (E/1.0e9) ** 0.333
    return D 

# f = func_Q(1.0e2, 1.0e11)/func_D(1.0e11) # eV^-1 cm^-3 
# print(f)

filename = 'plot_data_flux_p_AMS.dat'
Ea, jE_AMS, err_Ea, err_jE_AMS = np.loadtxt(filename, unpack=True, usecols=[0,1,2,3])

fig = plt.figure(figsize = (10, 8))
ax = plt.subplot(111)

n = 2.7
ax.errorbar(Ea, Ea**n*jE_AMS, Ea**n*err_jE_AMS, err_Ea, 'g^', markersize = 5.0, elinewidth = 2.5)     
# Đồ thị từ data

H = 10232e22 # Halo(cm)
h = 4.62e20  # Disk(cm)
u0 = 16e5    # Galatic wind(cm/s)

E = np.logspace(9.0,11.0,100)
mp = 938.272e6 # eV                                  #khối lượng proton

Vp = np.sqrt((E + mp)**2-mp**2) * 3.0e10 / (E + mp)     #vận tốc

#Note: Để tìm giá trị Q0, chọn thử giá trị bất kì sao cho ở mức năng lượng cao, 2 đồ thị dưới đây gần như trùng với đồ thị từ data (Vd chọn Q0 = 1.0e-37)

j1 = (Vp / (np.pi*4.0)) * (h/u0) * func_Q(1.0e-37, E) * (1.0 - np.exp((-u0*H) / func_D(E)))    # With Galatic wind 
j2 = (Vp / (np.pi*4.0)) * H * h * func_Q(1.0e-37, E) / func_D(E)                               # Without Galatic wind

#Vẽ đồ thị
ax.plot(E, E**n*j1, 'r--')   #màu đỏ
ax.plot(E, E**n*j2, 'b--')   #màu xanh dương

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()

ax.set_xlabel('E (eV)', fontsize=fs)

ax.set_ylabel('j(E) (eV^{-1} cm^{-2} s^{-1} s^{-1})', fontsize=fs)

for label_axf in (ax.get_xticklabels() + ax.get_yticklabels()):

    label_axf.set_fontsize(fs)

ax.set_xlim(1.0e9, 1.0e12)

#ax.set_ylim(jE_lim[0],jE_lim[1])

ax.legend(loc='lower left', prop={"size":22})

ax.grid(linestyle='--')


plt.show()

plt.savefig('fg_jE_p.png')