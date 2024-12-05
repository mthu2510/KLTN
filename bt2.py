import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import*
#mpl.rc("text",usetex=True)
fs=22
mp = 938.272e6 #eV                                  #khối lượng proton

#Định nghĩa hàm Q(Q0,E) và D(E)
def func_Q(Q0, E):

    Q = Q0*(E/1.0e9)**-2.4
    return Q

def func_D(E):                                      #Diffusion coefficient

    D = 1.0e28*(E/1.0e9)**0.333
    return D 


#Định nghĩa hàm j (With Galatic Wind)
def func_j(E,Q0,u0):
    H  = 4000.0*3.086e18    #Halo(cm)
    h  = 4.62e20            #Disk(cm)
    Vp = np.sqrt((E + mp)**2-mp**2)*3.0e10/(E + mp)                                #vận tốc tương đối tính

    
    j = (Vp/(np.pi*4.0))*(h/u0)*func_Q(Q0, E)*(1.0 - np.exp((-u0*H)/func_D(E)))    #With Galatic wind 
    #print(u0*H/func_D(E))
    return j


# ______________________________________________________________________________________________________

#Đồ thị từ data:
filename='plot_data_flux_p_AMS.dat'
Ea, jE_AMS, err_Ea, err_jE_AMS=np.loadtxt(filename,unpack=True,usecols=[0,1,2,3])

fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

n=2.7
ax.errorbar(Ea,Ea**n*jE_AMS,Ea**n*err_jE_AMS,err_Ea,'g^',markersize=5.0,elinewidth=2.5)     #Vẽ đồ thị từ data

#_________________________________________________________________________________________________________________________

#Thông số:
E=np.logspace(9.0,12.0,100)

Q0 = 1.0e-33

#_________________________________________________________________________________________________________________________

#Định nghĩa hàm mật độ dòng:
#j2 = (Vp/(np.pi*4.0))*H*h*func_Q(Q0,E)/func_D(E)                                

#Vẽ đồ thị
ax.plot(E,E**n*func_j(E,Q0,1e5),'r--',label='u0=1 km/s')        #red             
ax.plot(E,E**n*func_j(E,Q0,10e5),'g--',label='u0=10 km/s')      #green              
ax.plot(E,E**n*func_j(E,Q0,100e5),'b--',label='u0=100 km/s')    #blue            


ax.plot(E,E**n*func_j(E,Q0,0.5),'k--',label='u0=0.5 km/s')      #black      Without Galatic wind 

#_________________________________________________________________________________________________________________________

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


plt.show()
#import os
#directory = r"C:/Users/LAPTOP/Downloads/thesis"
#directory = os.getcwd()
#file_path = os.path.join(directory, "fg_jE_p.png")
#print(file_path)
# Create directory if it does not exist
#if not os.path.exists(directory):
 #   os.makedirs(directory)
#plt.savefig(r"C:/Users/LAPTOP/Downloads/thesis/fg_jE.png")
# Save the figure
#plt.savefig(file_path)
plt.savefig('fg_jE_p.png')
