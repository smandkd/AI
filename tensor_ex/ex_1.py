#%%
import tensorflow as tf
# %%
x1 = tf.constant(5)
x2 = tf.constant(3)
x3 = tf.constant(2)

result = tf.subtract(tf.multiply(x1, x2), tf.add(x2, x3))
print(f'result : {result}')
# %%
scalar_A = tf.constant(1)
scalar_B = tf.constant(1) 
lar_AB = scalar_A + scalar_B

vector_A = tf.constant([1, 2, 3])
vector_B = tf.constant([3, 5, 7])
dd_vector_AB = vector_A + vector_B

matrix_A = tf.constant([[1, 2], [3, 4]])
matrix_B = tf.constant([[4, 3], [2, 1]])

matrix_AB = tf.matmul(matrix_A, matrix_B)
print(dd_vector_AB)

# %%
x_data =[[1.],[2.],[3.]]
y_data =[[1.],[2.],[3.]]
print(x_data)
print(y_data)
# %%
W = tf.Variable(tf.random.normal((1, 1), mean=0, stddev=1))
b = tf.Variable(tf.random.normal((1, 1), mean=0, stddev=1))
# %%
print(W)
# %%
for j in range(len(x_data)):
    WX = tf.matmul([x_data[j]], W) 
    y_hat = tf.add(W, b)
    print(f'{y_data[j]} {y_hat}')
# %%
