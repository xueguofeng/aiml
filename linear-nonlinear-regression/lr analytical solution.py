import matplotlib.pyplot as plt
import numpy as np

x = [55, 71, 68, 87, 101, 87, 75, 78, 93, 73]
y = [91, 101, 87, 109, 129, 98, 95, 101, 104, 93]
plt.scatter(x, y)

def ols_algebra(x,y):    # using algebraic formula
    n=len(x)
    w1=(n*sum(x*y)-sum(x)*sum(y))/(n*sum(x*x)-sum(x)*sum(x))
    w0=(sum(x*x)*sum(y)-sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x))
    return w0,w1

def ols_matrix(x,y):      # using matrices
    w=(x.T*x).I*x.T*y
    return w

# calculate w0 and w1 using algebraic formula
w0,w1= ols_algebra( np.array(x), np.array(y) )
print("Using Algebra:", end=" ")
print(w0,w1)

'''
44.25604341391219 0.7175629008386778
'''


x = np.array(x).reshape(len(x),1)
x = np.concatenate((np.ones_like(x),x),axis=1)
x = np.matrix(x)
y = np.array(y).reshape(len(y),1)
y = np.matrix(y)
result = ols_matrix(x,y)
print("Using Matrix:", end=" ")
print(result)
'''
[ [44.25604341] [ 0.7175629 ] ]
'''

plt.plot(np.array([50,110]),np.array([50,110])*0.718+44.256,'r')
plt.show()