import random
import matplotlib.pyplot as plt

# model: y = w2 * x ** 2 + w1 * x + w0

x_data, y_data = [], []

w2_real = 1
w1_real = -5
w0_real = -2

# generate the data set
for tmpX in range(-10,20):
    noise = random.random() * 25
    tmpY = w2_real * tmpX ** 2 + w1_real * tmpX + w0_real + noise
    x_data.append(tmpX)
    y_data.append(tmpY)


lr = 0.00003  # learning rate
w2_learnt,w1_learnt,w0_learnt = 0,0,0  # initial w
iter = 200000  # iteration number

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

        d_loss_d_w0 += 2 * (w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt - y_real)
        d_loss_d_w1 += 2 * (w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt - y_real) * tmpX
        d_loss_d_w2 += 2 * (w2_learnt * tmpX ** 2 + w1_learnt * tmpX + w0_learnt - y_real) * tmpX ** 2

    d_loss_d_w0 = d_loss_d_w0 / len(x_data)
    d_loss_d_w1 = d_loss_d_w1 / len(x_data)
    d_loss_d_w2 = d_loss_d_w2 / len(x_data)

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

