'''
Created on Oct 19, 2010

@author: Peter
'''
import numpy as np

def loadDataSet():
    postingList = [[ 'my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) # union of two sets
    return list(vocabSet)  # set to list

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def RXtrainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) # calculate the p(c1)
                                                      # , p(c0) = 1 - p(c1)

    p0Num = np.zeros(numWords) # [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]
    p1Num = np.zeros(numWords) # [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]

    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # Abuse
            p1Num += trainMatrix[i]         # vector + vector ...
            p1Denom += sum(trainMatrix[i])  # how many words
        else:                      # Not Abuse
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = p1Num / p1Denom  # p(0|c1) p(1|c1) ... p(31|c1)
    p0Vect = p0Num / p0Denom  # p(0|c0) p(1|c0) ... p(31|c0)

    print(p0Num)
    print(p0Vect)

    print(p1Num)
    print(p1Vect)

    return p0Vect, p1Vect, pAbusive

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = np.ones(numWords) # [ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ]
    p1Num = np.ones(numWords) # [ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ]

    p0Denom = 2.0
    p1Denom = 2.0                        #change to 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    temp1 = p1Num/p1Denom
    temp0 = p0Num/p0Denom
    p1Vect = np.log( p1Num/p1Denom )          #change to np.log()
    p0Vect = np.log( p0Num/p0Denom )          #change to np.log()

    return p0Vect, p1Vect, pAbusive

#          word vector (bunch of 0 and 1),  p(0~31|c0), p(0~31|c1)  p(c1)
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):


  # dot product part
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #  ( Ln ( p( 0~31 |c1) )T . w +  ln( p(c1) )
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)  #  ( Ln ( p( 0~31 |c0) )T . w +  ln( p(c0) )
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array( setOfWords2Vec(myVocabList, testEntry) )  # sentence to vector
    temp = classifyNB(thisDoc, p0V, p1V, pAb)
    print(testEntry, 'classified as: ', temp)

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array( setOfWords2Vec(myVocabList, testEntry) )  # sentence to vector
    temp = classifyNB(thisDoc, p0V, p1V, pAb)
    print(testEntry, 'classified as: ', temp)

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():

    docList = [] # two dimensions, [ [ ], [ ], [ ] ]
    classList = [] #
    fullText = [] # 1 dimension, to keep all words

    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)# create vocabulary, 692 unique words

    trainingSet = list( range(50) )
    testSet = []           #create test set

    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #del(list(trainingSet)[randIndex])
        del trainingSet[randIndex]
    # split into trainingset(40 indices) and testset( 10 indices)



    trainMat = []  # 40 x 692, #word vector
    trainClasses = []  # 40 x 1

    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append( bagOfWords2VecMN (vocabList, docList[docIndex]) )
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0( np.array(trainMat), np.array(trainClasses) )
    # pOv,  p( 0~691 | 0)
    # p1v,  p( 0~691 | 0)
    # pSpam, p(1)
    #        p(0) = 1-p(1)

    errorCount = 0
# the above are all training work, the following are all testing
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])

    print('the error rate is: ', float(errorCount)/len(testSet))
    #return vocabList, fullText

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []           #create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
