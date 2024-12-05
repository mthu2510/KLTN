import jax.numpy as jnp
from jax import grad 
import jax.random as jr 
import matplotlib.pyplot as plt
from jax import jit

seed = 42
key = jr.PRNGKey(seed)
 
def func_lin(pars, x):
    return pars[0]*x + pars[1] # ax + b = y

def func_loss(pars,x,y):
    predictions = func_lin(pars,x)
    return jnp.mean((predictions - y) ** 2) # 1 dạng sai số cho y để hiệu chỉnh a và b 

@jit 
# Hàm update là hàm hiệu chỉnh a và b 
def update(pars, x, y, lr = 0.1):  # lr: learning rate 
    grads = grad(func_loss)(pars,x,y) # lấy đạo hàm của func_loss 
    return pars - lr*grads 

x = jnp.linspace(0.0,1.0,20) 
dyr = jr.normal(key, (len(x), )) * 0.2 # lấy random 

pars = jnp.array([2.5,8.0]) # số liệu đúng cho a = 2.5 và b = 8 
y = func_lin(pars,x) + dyr 

N_epoch = 1000
pars_scan = jnp.array([1000.0,-200.0]) # số liệu input cho a và b 

update_values = [] # Ghi lại giá trị mỗi lần update
update_func_loss = []
update_func_lin = []

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

for epoch in range(N_epoch):
    pars_scan = update(pars_scan, x, y)
    
    if epoch % 100 == 0: 
        print(epoch + 1, pars_scan) 
        ax.plot(x, func_lin(pars_scan, x), '-')
    
    update_values.append(pars_scan)
    update_func_loss.append(func_loss(pars_scan, x, y))
    update_func_lin.append(func_lin(pars_scan, x))

ax.plot(x,y,'ko')
ax.plot(x,func_lin(pars_scan, x),'r-')
ax.set_title('Sự thay đổi của hàm sau mỗi lần update')

ax2.plot(update_func_loss, linestyle = 'dotted')
ax2.set_title('Sự thay đổi của hàm loss sau mỗi lần update')

plt.show() # type: ignore

with open('JAX.txt','w') as JAX:
        s0 = '{0:^5} {delim} {1:^12} {delim} {2:^12} {delim} {3:^12} {delim} {4:^12} \n'
        JAX.write(s0.format('N', 'a', 'b', 'func_loss', 'func_lin', delim = '|'))
        for i in range (N_epoch):
            JAX.write(f' {i:5} | {update_values[i][0]:10.5f} | {update_values[i][1]:10.5f} | {update_func_loss[i]:10.4f} | {update_func_lin[i]} \n')

# Để vẽ được sự thay đổi của func_loss theo mỗi lần update thì tạo 1 subplot, vẽ trên hình khác với x = N_epoch và y = func_loss 