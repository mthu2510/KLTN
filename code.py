import jax.numpy as jnp
from jax import grad 
import jax.random as jr 
import matplotlib.pyplot as plt
from jax import jit

seed = 42
key = jr.PRNGKey(seed)


def func_lin(pars, x):
    return pars[0]*x + pars[1] # ax + b = 0

def func_loss(pars,x,y):
    predictions = func_lin(pars,x)
    return jnp.mean((predictions - y) ** 2)

@jit 
def update(pars, x, y, lr = 0.1):
    grads = grad(func_loss)(pars,x,y)
    return pars - lr*grads 

x = jnp.linspace(0.0,1.0,20)
dyr = jr.normal(key, (len(x), )) * 0.2

pars = jnp.array([2.5,8.0])
y = func_lin(pars,x) + dyr 

N_epoch = 1000
pars_scan = jnp.array([1000.0,-200.0]) 
for epoch in range(N_epoch):
    pars_scan = update(pars_scan, x, y)
    if epoch % 100 == 0:
        print(epoch + 1, pars_scan) 
        plt.plot(x,func_lin(pars_scan, x),'-')

plt.plot(x,y,'ko')
plt.plot(x,func_lin(pars_scan, x),'r-')
plt.show() # type: ignore