import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'labels']
    data = np.array(df.iloc[:, [0, 1, 2, 3, -1]])
    # split old features and new features
    return data[:, :4], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class LogisticRegressionClassifier:
    def __init__(self, max_iter=200, lr=0.01):  # constructor
        self.max_iter = max_iter  # define a member variable - iteration number
        self.lr = lr  # define a class a member variable -  learning rate

    def sigmoid(self, X):   #
        return 1 / (1 + np.exp(-X))  # y = Sigmod(x) =  1 / ( 1+ e^(-x) )

    def data_matrix(self, X):
        data_mat = []   # this is a temp variable , not the member variable
        for x in X:     # features (x1,x2,x3,x3)
            data_mat.append([1, * x])  # add x0 using Python List * operator
        return data_mat # features (x0,x1,x2,x3,x3)

    def softmax(self, d):  # 多分类用 softmax
        return np.exp(d) / np.sum(np.exp(d))

    def fit(self, X, y):
        data_mat = self.data_matrix(X)
        self.weights = np.zeros((len(data_mat[0]), 3), dtype=np.float32)
        # Parameters, (5x3)
        # w10 w20 w30
        # w11 w21 w31
        # w12 w22 w32
        # w13 w23 w33
        # w14 w24 w34

        for step_ in range(self.max_iter): # iteration

            for i in range(len(data_mat)):

                # pre = self.sigmoid(np.dot(data_mat[i], self.weights))
                # pre = self.softmax(np.dot(data_mat[i], self.weights))

                tempX = data_mat[i]
                # 5x1

                tempZ = np.dot( tempX, self.weights )  #  Θ X
                #  3x5 5x1 = 3x1

                pre = self.softmax( tempZ )   # Y = Softmax (Θ X)
                # 3x1 = softmax ( 3x1 )

                obj = np.eye(3)[ int( y[i] ) ] # transform a label value to one-hot
                # [0] -> [1 0 0], [1] -> [0 1 0], [2] -> [0 0 1]

                err = pre - obj
                # 3x1 = 3x1 - 3x1
                                 #  I{y=0} - P { y=0 | x,  Θ}  , for dl/dΘ0
                                 #  I{y=1} - P { y=1 | x,  Θ}  , for dl/dΘ1
                                 #  I{y=2} - P { y=2 | x,  Θ}  , for dl/dΘ2
                tmp2 = np.transpose([data_mat[i]])
                tmp1 = tmp2 * err
                self.weights = self.weights - self.lr * tmp1
                # tmp1 = {ndarray: (5, 3)} [[ 0.33333333  0.33333333 -0.66666667], [ 2.56666667  2.56666667 -5.13333333], [ 1.26666667  1.26666667 -2.53333333], [ 2.23333333  2.23333333 -4.46666667], [ 0.73333333  0.73333333 -1.46666667]]...View as Arrayself.weights =  self.weights - self.lr * np.transpose([data_mat[i]]) * err
                #               3x5 = 3x5 - 3x1 1x5

            if (step_ % 1 == 0):
                print("*********************************************************")
                print("round {} ,score {}".format(step_,  self.score(X_test, y_test)))

    '''
                if (step_ % 1 == 0):
                    print("*********************************************************")
                    print("round {}\nweights\n {} \nerr {} \nscore {}".format(
                        step_,self.weights, err,self.score(X_test, y_test)))
                    print("distribution\t", pre)
    '''

    def score(self, X, y):
        X = self.data_matrix(X)
        right = 0
        for i in range(len(X)):
            pre = np.dot(X[i], self.weights)
            # 3x1 =  3x5 5x1

            pre2 = np.argmax(pre)  # return the index of max value
            if pre2 == y[i]:  # if predict correctly
                right += 1
        return right / len(X)


lrc = LogisticRegressionClassifier(max_iter=500)

# Training
lrc.fit(X_train, y_train)

