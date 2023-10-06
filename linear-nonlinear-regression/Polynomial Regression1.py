import random
import matplotlib.pyplot as plt

x_data, y_data = [], []

# model: y = w2 * x ** 2 + w1 * x + w0
w2_real = 5
w1_real = -10
w0_real = -200

# generate the data set: using the model, and adding noises
for tmpX in range(-20,20):
    noise = random.random() * 10
    tmpY = w2_real * tmpX ** 2 + w1_real * tmpX + w0_real + noise
    x_data.append(tmpX)
    y_data.append(tmpY)

# by observing the samples, we need to decide the model and how many parameters - "Feature Engineering"
# with the traditional ML approach, we have to do the job manually based on our experiences


lr = 0.00003  # learning rate
w2_learnt,w1_learnt,w0_learnt = 0,0,0  # initial w
iter = 50000 # iteration number

# calculate the loss using mse
def loss_function_mse():
    total_loss = 0
    for i in range(0, len(x_data)):
        tmpX = x_data[i]
        y_real = y_data[i]
        y_hat = w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt
        total_loss += (y_hat - y_real) ** 2
    return total_loss / len(x_data)


for i in range(0, iter):
    d_loss_d_w0 = 0
    d_loss_d_w1 = 0
    d_loss_d_w2 = 0

    for j in range(0, len(x_data)):
        tmpX = x_data[j]
        y_real = y_data[j]

        # the error from each sample - (xi, yi),  (y_hat - y_real) ** 2
        # for each sample, calculate the partial derivatives by using Xi, parameters and Yi
        d_loss_d_w0 += 2 * (w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt - y_real)
        d_loss_d_w1 += 2 * (w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt - y_real) * tmpX
        d_loss_d_w2 += 2 * (w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt - y_real) * tmpX ** 2

    # Get the partial derivatives for the total loss/error, which is the sum of all sample
    d_loss_d_w0 = d_loss_d_w0 / len(x_data)
    d_loss_d_w1 = d_loss_d_w1 / len(x_data)
    d_loss_d_w2 = d_loss_d_w2 / len(x_data)

    # update the parameters
    w0_learnt = w0_learnt - lr * d_loss_d_w0
    w1_learnt = w1_learnt - lr * d_loss_d_w1
    w2_learnt = w2_learnt - lr * d_loss_d_w2

    if i % 10000 ==0:
        print("current loss: ",loss_function_mse()," iter num:", i)


plt.plot(x_data,y_data,'b')


y_hat = [ w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt for tmpX in  x_data ]
plt.plot(x_data,y_hat,'r')
print(w2_learnt,w1_learnt,w0_learnt)

plt.show()

