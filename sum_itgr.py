import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0, 100, 100)
m = np.arange(-10, 11, 1)

x = np.linspace(-10, 10, 10000)
dx = np.append(np.diff(x),0)

A_n = np.sum(n[:,np.newaxis]**2 * m[np.newaxis,:]**3, axis=1)

B_n = np.sum(n[:,np.newaxis]**2 * x[np.newaxis,:]**2 * dx[np.newaxis,:], axis=1)

C_n = np.sum(np.exp(-n[:,np.newaxis]) * np.exp(x[np.newaxis,:]) * x[np.newaxis,:] * dx[np.newaxis,:], axis = 1)

print(A_n[0:20])
print(B_n[0:20])
print(C_n[0:20])