import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rc("text",usetex=True)

fs=22

def func_Q(Q0, E):
    Q = Q0*(E/1.0e9)**-2.4
    return Q

def func_D(E):
    D = 1.0e28*(E/1.0e9)**0.333
    return D 
f = func_Q(1.0e2, 1.0e11)/func_D(1.0e11) # eV^-1 cm^-3 
print(f)


filename='plot_data_flux_p_AMS.dat'
Ea, jE_AMS, err_Ea, err_jE_AMS=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3])

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

n=2.7
ax.errorbar(Ea,Ea**n*jE_AMS,Ea**n*err_jE_AMS,err_Ea,'g^',markersize=5.0,elinewidth=2.5)

E=np.logspace(9.0,11.0,100)
ax.plot(E,E**n*func_Q(1.0e41,E)/func_D(E),'r--')

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()

ax.set_xlabel('E (eV)',fontsize=fs)

ax.set_ylabel('j(E) (eV^{-1} cm^{-2} s^{-1} s^{-1})',fontsize=fs)

for label_axf in (ax.get_xticklabels() + ax.get_yticklabels()):

    label_axf.set_fontsize(fs)

ax.set_xlim(1.0e9,1.0e12)

#ax.set_ylim(jE_lim[0],jE_lim[1])

ax.legend(loc='lower left', prop={"size":22})

ax.grid(linestyle='--')


plt.savefig("fg_jE_p.png")