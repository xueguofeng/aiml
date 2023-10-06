import logRegres
import numpy as np

dataArr,labelMat = logRegres.loadDataSet()

weights = logRegres.gradAscent(dataArr,labelMat)
print(weights)
temp = weights.getA() # return self as an ndarray object
logRegres.plotBestFit(temp)

weights = logRegres.stocGradAscent0( np.array(dataArr),labelMat )
print(weights)
logRegres.plotBestFit(weights)

weights = logRegres.stocGradAscent1( np.array(dataArr),labelMat )
print(weights)
logRegres.plotBestFit(weights)

print("The end")