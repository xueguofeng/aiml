import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


##### tensorflow variable
my_tensor = tf.constant([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

my_variable = tf.Variable(my_tensor)

my_Numpy_Variable = my_variable.numpy()

print(my_variable)
my_variable.assign([[1.1, 2.1, 3.1], [3.1, 4.1, 5.1]])
print(my_variable)

##### tensorflow matmul
a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b) # 2x3 x 3x2 = 2x2
print(c)


##### tensorflow gradient: y = x ^ 2， y' while x=3

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x)
print(dy_dx)

##### tensorflow gradient: y = x ^ 3， y' while x=3

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x * x
dy_dx = g.gradient(y, x)
print(dy_dx)

##### tensorflow gradient: y = x ^ 2， y'' while x=3

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  with tf.GradientTape() as gg:
    gg.watch(x)
    y = x * x
  dy_dx = gg.gradient(y, x)      # y’ = 2*x = 2*3 =6
d2y_dx2 = g.gradient(dy_dx, x)  # y’’ = 2
print(d2y_dx2)


##### tensorflow gradient: y = 5 * (x0**4) + 4* (x1**3) + 3 * (x2**2) + 2 * (x3) + 100
# dy/dx0, dy/dx1, dy/dx2, dy/dx3

x0 = tf.Variable(1.0, name='x0')
x1 = tf.Variable(2.0, name='x1')
x2 = tf.Variable(3.0, name='x2')
x3 = tf.constant(4.0, name='x3')

with tf.GradientTape() as g:
    g.watch(x0)
    g.watch(x1)
    g.watch(x2)
    g.watch(x3)
    y = 5 * (x0**4) + 4* (x1**4) + 3 * (x2**2) + 2 * (x3) + 100
grad = g.gradient(y, [x0, x1, x2, x3])
for temp in grad:
  print(temp)


# mean
x = tf.constant([[1., 1.], [2., 2.]])
temp = tf.reduce_mean(x)
print(temp)

temp = tf.reduce_mean(x, 0) # mean for each col
print(temp)

temp = tf.reduce_mean(x, 1) # mean for each row
print(temp)

temp = tf.math.reduce_mean(x, 1) # mean for each row
print(temp)

print("end")




'''
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x)

print(dy_dx)

'''