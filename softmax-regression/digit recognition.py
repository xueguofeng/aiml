import pickle
import gzip

from matplotlib import pyplot
import numpy as np

with gzip.open("mnist.pkl.gz", "rb") as f:
        ((train_X, train_y), (test_X, test_y), _) = pickle.load(f, encoding="latin-1")


fig = pyplot.figure(figsize=(10,7))
for i in range(15):
    ax = fig.add_subplot(3, 5, i+1)
    ax.imshow( train_X[i].reshape((28, 28)), cmap=pyplot.get_cmap('gray') )
    ax.set_title('Label (y): {y}'.format(y=train_y[i]))
    pyplot.axis('off')
pyplot.show()


def one_hot(y, c):
    # y--> label/ground truth.
    # c--> Number of classes.

    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c))

    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1

    return y_hot


def softmax(z):
    # z--> linear part.

    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))

    # Calculating softmax for all examples.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])

    return exp


def fit(X, y, lr, c, epochs):
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.

    # m-> number of training examples
    # n-> number of features
    m, n = X.shape

    # Initializing weights and bias randomly.
    w = np.random.random((n, c))
    b = np.random.random(c)
    # Empty list to store losses.
    losses = []

    # Training loop.
    for epoch in range(epochs):

        # Calculating hypothesis/prediction.
        z = X @ w + b     # 50000x10 = 50000x784 x 784x10 + 50000x10[b1,b2...b10]
        y_hat = softmax(z)  # 50000x10

        # One-hot encoding y.
        y_hot = one_hot(y, c) # 50000x10

        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1 / m) * np.dot(X.T, (y_hat - y_hot))
        b_grad = (1 / m) * np.sum(y_hat - y_hot)

        # Updating the parameters:   (I-P) = (y_hot - y_hat) = - (y_hat - y_hot)
        w = w - lr * w_grad
        b = b - lr * b_grad

        # Calculating loss and appending it in the list.
        loss = - np.mean(   np.log(y_hat[ np.arange(len(y)), y] )   )


        tmp3 = np.arange(len(y))
        tmp2 = y_hat[tmp3,y]
        tmp1 = np.log(tmp2)
        loss = - np.mean( tmp1 )


        losses.append(loss)
        # Printing out the loss at every 100th iteration.
        if epoch % 10 == 0:
            print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=epoch, loss=loss))

    return w, b, losses


# Training
w, b, l = fit(train_X, train_y, lr=0.9, c=10, epochs=200)


def predict(X, w, b):
    # X --> Input.
    # w --> weights.
    # b --> bias.

    # Predicting
    z = X @ w + b
    y_hat = softmax(z)

    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)

train_preds = predict(train_X, w, b)
result = accuracy(train_y, train_preds)
print("Accuracy for training data: ",result)

test_preds = predict(test_X, w, b)
result = accuracy(test_y, test_preds)
print("Accuracy for testing data: ",result)

fig = pyplot.figure(figsize=(15, 10))
for i in range(40):
    ax = fig.add_subplot(5, 8, i + 1)
    ax.imshow( test_X[i].reshape((28, 28)), cmap=pyplot.get_cmap('gray') )
    ax.set_title('y: {y}/ y_hat: {y_hat}'.format(y=test_y[i], y_hat=test_preds[i]))
    pyplot.axis('off')

pyplot.show()

