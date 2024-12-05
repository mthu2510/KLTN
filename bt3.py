import numpy as np 

def func_rhs(y,t): 
    return t 

t= np.linspace(0.0,1.0,10001)
dt=t[1]-t[0]

y= np.zeros_like(t)
y_rk4= np.zeros_like(t)

for i in range(len(t)-1):
    k1= func_rhs(y[i],t[i])
    k2= func_rhs(y[i]+k1*dt/2.0,t[i]+dt/2.0)
    y[i+1]=y[i]+t[i]*dt

y_ana=t**2/2.0
y_num=np.cumsum(t*dt)
print(y_ana[-1])
print(y[-1])
print(y_num[-1])