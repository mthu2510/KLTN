import jax.numpy as jnp
from jax import grad 
import jax.random as jr 
import matplotlib.pyplot as plt
from jax import jit

seed = 42
key = jr.PRNGKey(seed)

# Q0 = pars[0], alpha = pars[1]

H = 10232e22 #Halo(cm)
h = 4.62e20  #Disk(cm)
u0 = 16e5    #Galatic wind(cm/s)

def func_Q (pars, E):
    Q = pars[0] * (E*1e-9) ** pars[1]
    return Q

def func_D (E):
    D = 1.0e28 * (E / 1.0e9) ** 0.333
    return D 
 
def func_f (pars, E):
    f = (func_Q (pars, E) * h / u0) * (1 - jnp.exp(- u0 * H / func_D (E)))
    return f

def func_loss(pars, x, y):
    predictions = func_f(pars, x)
    return jnp.mean((predictions - y) ** 2)


@jit 
def update(pars, x, y, lr):  
    grads = grad(func_loss) (pars, x, y) 
    return pars - lr * grads 

x = jnp.logspace(9, 12, 20)  # Giá trị E từ 1 đến 1000 (eV) trên thang log
dyr = jr.normal(key, (len(x), )) * 0.05

pars = jnp.array([10.0, -2.0]) 
y = func_f(pars,x) + dyr 

N_epoch = 100
pars_scan = jnp.array([5.0, -1.5])

grads_init = grad(func_loss) (pars_scan, x, y) 
lr = 0.05 * pars_scan / jnp.abs(grads_init)

update_values = [] 
update_func_loss = []
update_f = []

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

for epoch in range(N_epoch + 1):
    pars_scan = update(pars_scan, x, y, lr)
    
    if epoch % 10 == 0: 
        print(epoch + 1, pars_scan) 
        ax.plot(x, func_f(pars_scan, x), '-')
    
    update_values.append(pars_scan)
    update_func_loss.append(func_loss(pars_scan, x, y))
    update_f.append(func_f(pars_scan, x))

ax.plot(x, y, 'ko')
ax.plot(x, func_f(pars_scan, x), 'r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Sự thay đổi của hàm sau mỗi lần update')

ax2.set_yscale('log')
ax2.plot(update_func_loss, linestyle = 'dotted')
ax2.set_title('Sự thay đổi của hàm loss sau mỗi lần update')

plt.show()