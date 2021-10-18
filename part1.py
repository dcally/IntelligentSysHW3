import numpy as np
import matplotlib.pyplot as plt
import random


def setCreator():
    train_dataImg = np.loadtxt("MNISTnumImages5000_balanced.txt")
    train_dataLabel = np.loadtxt("MNISTnumLabels5000_balanced.txt")
    zero = []
    one = []
    seven = []
    nine = []
    interator = 0
    # print(train_dataImg.shape)
    for num in train_dataImg:
        # (train_dataImg[interator])
        if train_dataLabel[interator] == 0:
            zero.append(train_dataImg[interator])
        if train_dataLabel[interator] == 1:
            one.append(train_dataImg[interator])
        if train_dataLabel[interator] == 7:
            seven.append(train_dataImg[interator])
        if train_dataLabel[interator] == 9:
            nine.append(train_dataImg[interator])
        interator = interator + 1

    print(interator)
    random.shuffle(zero)
    random.shuffle(one)
    random.shuffle(seven)
    random.shuffle(nine)
    np.savetxt('zeroSet.txt', zero, fmt='%1.4f', delimiter=',')
    np.savetxt('oneSet.txt', one, fmt='%1.4f', delimiter=',')
    np.savetxt('sevenSet.txt', seven, fmt='%1.4f', delimiter=',')
    np.savetxt('nineSet.txt', nine, fmt='%1.4f', delimiter=',')

def CreateTrainSet():
    totalIterator = 0
    oneCount = 0
    zeroCount = 0
    oneSet = np.loadtxt("oneSet.txt", delimiter=",")
    zeroSet = np.loadtxt("zeroSet.txt", delimiter=",")
    trainLabels = []
    testLabels = []
    testDataSet = []
    trainDataSet = []
    for num in range(1000):
        zeroOrOne = np.random.randint(2)
        if oneCount == 500:
            zeroOrOne = 0
        elif zeroCount == 500:
            zeroOrOne = 1
        # print(zeroOrOne)
        if zeroOrOne == 0:
            if totalIterator < 200:
                testLabels.append(zeroOrOne)
                testDataSet.append(zeroSet[zeroCount])
            else:
                trainLabels.append(zeroOrOne)
                trainDataSet.append(zeroSet[zeroCount])
            zeroCount = zeroCount + 1
            #print(f'Zero Count" + {str(zeroCount)}')
        else:
            if totalIterator < 200:
                testLabels.append(zeroOrOne)
                testDataSet.append(oneSet[oneCount])
                #print(f'testDataSet + {str(oneSet[oneCount])}')
            else:
                trainLabels.append(zeroOrOne)
                trainDataSet.append(oneSet[oneCount])
                #print(f'trainDataSet + {str(oneSet[oneCount])}')
            oneCount = oneCount + 1
            #print(f'One Count" + {str(oneCount)}')

        totalIterator = totalIterator + 1
    #print(totalIterator)
    np.savetxt('testDataSet.txt', testDataSet, fmt='%1.4f', delimiter=',')
    np.savetxt('trainDataSet.txt', trainDataSet, fmt='%1.4f', delimiter=',')
    np.savetxt('trainLabels.txt', trainLabels, fmt='%i', delimiter=',')
    np.savetxt('testLabels.txt', testLabels, fmt='%i', delimiter=',')

def CreateChallengeSet():
    totalIterator = 0
    sevenCount = 0
    nineCount = 0
    sevenSet = np.loadtxt("sevenSet.txt", delimiter=",")
    nineSet = np.loadtxt("nineSet.txt", delimiter=",")
    challengeLabels = []
    challengeDataSet = []
    for num in range(200):
        zeroOrOne = np.random.randint(2)
        if sevenCount == 100:
            zeroOrOne = 1
        elif nineCount == 100:
            zeroOrOne = 0
        # A zero places a 7
        if zeroOrOne == 0:
            challengeLabels.append(7)
            challengeDataSet.append(sevenSet[sevenCount])
            sevenCount = sevenCount + 1
        # a One places a nine
        else:
            challengeLabels.append(9)
            challengeDataSet.append(nineSet[nineCount])
            nineCount = nineCount + 1

        totalIterator = totalIterator + 1
    print(totalIterator)
    print(nineCount)
    print(nineCount)
    np.savetxt('challengeDataSet.txt', challengeDataSet, fmt='%1.4f', delimiter=',')
    np.savetxt('challengeLabels.txt', challengeLabels, fmt='%i', delimiter=',')
