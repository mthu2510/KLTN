import numpy as np 
import matplotlib.pyplot as plt


t = np.linspace(0,np.pi,10)
a = np.zeros(len(t)) + 1.0    #y[0] = 1.0

def f(t,a):
    return a*np.sin(t)**2

#Step size
h = 1.0   #np.pi/10 

for i in range(0, len(t)-1):                   

    k1 = f(t[i]       , a[i]          )
    k2 = f(t[i] + h/2 , a[i] + h*k1/2 )
    k3 = f(t[i] + h/2 , a[i] + h*k2/2 )
    k4 = f(t[i] + h   , a[i] + h*k3   )
  
    t[i+1] = t[i] + h
    a[i+1] = a[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


print (a)                                   #N
print("-----------------------------")
print (np.exp(t/2 - np.sin(2*t)/4))         #A



#Plot Phương pháp số
plt.plot(t,a,'b--', label = "Numerical Method")

#Plot Phương pháp giải tích
x = t
y = np.exp(x/2 - np.sin(2*x)/4)
plt.plot(x,y,'r',label = "Analytical Method")


#Plot names
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend()

plt.show()









