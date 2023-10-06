import bayes

listOPosts,listClasses = bayes.loadDataSet() # return 6 sentences with their labels
myVocalList = bayes.createVocabList(listOPosts) # 32 unique words
print(myVocalList)

trainMat = []
for pstinDoc in listOPosts:
    temp = bayes.setOfWords2Vec(myVocalList,pstinDoc)
    print(temp) # the word vector for each sentence
    trainMat.append( temp )

p0V, p1V, pAb = bayes.RXtrainNB0(trainMat,listClasses)

print(p1V)  # p(0|c1) p(1|c1) ... p(31|c1)
print(p0V)  # p(0|c0) p(1|c0) ... p(31|c0)
print(pAb)  # p(c1)
print(1-pAb)  # p(c0)

p0V, p1V, pAb = bayes.trainNB0(trainMat,listClasses)
print(p0V)  # p(0|c1) p(1|c1) ... p(31|c1)
print(p1V)  # p(0|c0) p(1|c0) ... p(31|c0)
print(pAb)  # p(c1)
print(1-pAb)  # p(c0)