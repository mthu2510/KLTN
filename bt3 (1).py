import numpy as np

t=np.linspace(0,1,100)
dt = t[1]-t[0]
y = np.zeros(len(t)) + 1.5  #y[t = 0] = 1.5

for i in range(1,len(t)):
    y[i]=y[i-1]+t[i-1]*dt

print(y)
print('------------------')
print(t**2/2.0 + 1.5)