import matplotlib.pyplot as plt
import numpy as np

# generate and draw N samples (xi,yi)
xx = [55, 71, 68, 87, 101, 87, 75, 78, 93, 73]
yy = [91, 101, 87, 109, 129, 98, 95, 101, 104, 93]
plt.scatter(xx, yy)

# lr - learning rate
# number of iteration
def gradient_descent(x, y, lr, num_iter):
    w1 = 0  # Initial value, 0
    w0 = 0  # Initial value, 0

    for i in range(num_iter):


        ##### 正向传播：固定w0和w1，利用N组(xi, yi)，求MSE及梯度（dMSE / dw0和dMSE / dw1）。
        # Calculate the y_hat
        # x is a 10 x 1 vector, and w0 / w1 are a scalar, result is still a vector
        y_hat = (w1 * x) + w0   # original linear equation

        # Calculate the gradient of current location, which uses all the samples
        # For Mini Batch and Stochastic, only use part of the samples.
        w0_gradient = (-2/len(x)) * sum(y-y_hat)
        w1_gradient = (-2/len(x)) * sum(x*(y-y_hat))
        # numpy sum is adding all values in a vector into one number

        ##### 反向传播：根据求得的梯度（dMSE / dw0和dMSE / dw1），移动一小步，更新w0和w1。
        # To find the minimum, move at the opposite direction of gradient
        w1 = w1 - lr * w1_gradient
        w0 = w0 - lr * w0_gradient

    return w1, w0

w1, w0 = gradient_descent( np.array(xx), np.array(yy), lr=0.0001, num_iter=1000 )
print(w1,w0)

#               x                y = w1 x + w0
plt.plot( np.array([50, 110]), np.array([50, 110]) * w1 + w0, 'r')

plt.show()
