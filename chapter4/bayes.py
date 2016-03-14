def loadDataSet():
    postList=['my dog has flea problems help please',
              'maybe not take him to dog park stupid',
              'my dalmation is so cute I love him',
              'stop posting stupid worthless garbage',
              'mr licks ate my steak how to stop hime',
              'quit buying worthless dog food stupid']
    postingList = [post.split() for post in postList]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# one-hot word2vec
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print 'the word: %d is not in my vocabulary!'%word
    return returnVec

from numpy import *

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect, p1Vect, pAbusive


