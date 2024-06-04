#%%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
%matplotlib inline
# %%
def step_function(x):
    if x < thres:
        return 0
    else : 
        return 1

# %%
def gen_training_data(data_point):
    x1 = np.random.random(data_point)
    x2 = np.random.random(data_point)
    y = ( (x1 + x2) > 1 ).astype(int)
    training_set = [ ((x1[i], x2[i]), y[i]) for i in range(len(x1)) ]
    
    return training_set 
# %%
thres = 0.5
w = np.array([0.3, 0.9])
lr = 0.1
data_point = 100
epoch = 10
training_set = gen_training_data(data_point)
# %%
print(training_set[0:5])
# %%
plt.figure(0)
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 1.1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

for x,y in training_set:
    if y==1 :
        plt.plot(x[0], x[1], 'bo')
    else:
        plt.plot(x[0], x[1], 'go')

plt.show()
# %%
from time import sleep 
# %%
plt.figure(0)
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 1.1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

xx = np.linspace(0, 1, 50)
for i in range(epoch):
    cnt = 0
    for x, y in training_set:
        clear_output(wait=True)
        u = sum(x*w)
        error = y - step_function(u)
        
        for index, value in enumerate(x):
            w[index] = w[index] + lr*error*value
            
        for xs, ys in training_set[0:cnt]:
            plt.ylim(-0.1, 1.1)
            plt.xlim(-0.1, 1.1)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            
            if ys == 1:
                plt.plot(xs[0], xs[1], 'bo')
            else:
                plt.plot(xs[0], xs[1], 'go')
                    
        yy = -w[1]/w[0] * xx + thres/w[0]
        plt.plot(xx, yy)
        plt.show()
        cnt = cnt + 1
        sleep(0.01)
# %%
