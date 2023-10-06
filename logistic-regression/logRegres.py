import numpy as np

def loadDataSet():

    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        #     b    + w1 * x1 + w2 * x2
        # w0 * x0  + w1 * x1 + w2 * x2
        # x0 = 1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # adding the two decimals
        # target label, 1 or 0
        labelMat.append(int(lineArr[2])) # adding the 0 or 1

    return dataMat, labelMat  # return two lists

def sigmoid(Z):
    return 1.0 / (1 + np.exp(-Z))

def gradAscent(dataMatIn, classLabels):

    dataMatrix = np.mat(dataMatIn)             #convert from Python List to NumPy matrix
    labelMat = np.mat(classLabels).transpose() #convert from Python List to NumPy matrix
    m, n = np.shape(dataMatrix)  # M samples - 100, N features - 3

    alpha = 0.001  # learning rate
    maxCycles = 500   # number of iteration

    weights = np.ones((n, 1))  # initial weights (1,1,1)

    # one there is one billion data samples, we cannot use every single one 500 times. Here there is
        #only 100 samples, so no need for mini patch or stochastic
    for k in range(maxCycles):              #heavy on matrix operations
                                               #matrix mult,  z = w0 * 1 + w1 * x1 + w2 * x2
        p = sigmoid( dataMatrix * weights )    #              z =      b + w1 * x1 + w2 * x2
             # input:  100x3 * 3x1, 100 z
             # output: 100 sigmod values for 100 z

        error = (p - labelMat)
        #vector subtraction,  (p-yi)

        # computer calculate this differently, find all p-y, then multiply every p-y by every x
        weights = weights - alpha * dataMatrix.transpose() * error # matrix mult
                                         #  3x100       *    100x1
                                         # the result:  3x1

    return weights

def stocGradAscent0(dataMatrix, classLabels):

    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        p = sigmoid( sum( dataMatrix[i] * weights ) )
        # input:  z = x0 * w0 + x1 * w1 + x2 * w2
        # output: 1 sigmod value

        error = p - classLabels[i]  # 1 value,  (p-yi)
        weights = weights - alpha * error * dataMatrix[i]
                                   # the result:  3x1

    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):

    m, n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones

    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0, len(dataIndex)))#go to 0 because of the constant

            p = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = p - classLabels[randIndex]
            weights = weights - alpha * dataMatrix[randIndex] * error
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):

    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)

    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()





def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        