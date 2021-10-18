import numpy as np
import matplotlib.pyplot as plt
import math as ma


def NewTrainer():
    trainDataSet = np.loadtxt("trainDataSet.txt", delimiter=',')
    trainLabels = np.loadtxt("trainLabels.txt", delimiter=',')
    w = np.loadtxt("initialWeights.txt", delimiter=',')
    preW = w
    iterator = 0
    secInt = 0
    sValArr = []
    sVal = 0
    theta = 0
    numberOfEpochs = 60
    y = 0
    n = 0.01

    for values in range(numberOfEpochs):
        iterator = 0
        for num in trainDataSet:
            secInt = 0
            sVal = 0
            for ite in trainDataSet[iterator]:
                sVal = sVal + (w[secInt] * ite)
                secInt = secInt + 1

            secInt = 0
            # if sVal > theta:
            #     y = 1
            # else:
            #     y = 0
            y = trainLabels[iterator]
            preW = w
            for kachow in trainDataSet[iterator]:
                # if secInt > 0:
                #     print(f'Before {w[secInt]}')
                #     #w[secInt] = preW[secInt - 1] + ((n * y) * (kachow - preW[secInt - 1]))
                #     print(f'After {w[secInt]}')
                w[secInt] = preW[secInt] + ((n * kachow) * (y - preW[secInt]))
                secInt = secInt + 1
            sValArr.append(sVal)
            iterator = iterator + 1
    print(iterator)

    np.savetxt('trainedNewWeights.txt', w, fmt='%1.4f', delimiter=',')

def Newtesting():
    testDataSet = np.loadtxt("testDataSet.txt", delimiter=',')
    testLabels = np.loadtxt("testLabels.txt", delimiter=',')
    w = np.loadtxt("trainedNewWeights.txt", delimiter=',')
    trueNeg = 0
    truePos = 0
    falseNeg = 0
    falsePos = 0
    iterator = 0
    secInt = 0
    sVal = 0
    y = 0

    precisions = []
    recalls = []
    fScores = []
    sensitivity = []
    spec = []

    for theta in range(40):
        iterator = 0
        trueNeg = 0
        truePos = 0
        falseNeg = 0
        falsePos = 0
        for num in testDataSet:
            secInt = 0
            sVal = 0
            for ite in testDataSet[iterator]:
                sVal = sVal + (w[secInt] * ite)
                secInt = secInt + 1

            if sVal > theta:
                y = 1
                if y == int(testLabels[iterator]):
                    truePos += 1
                else:
                    falsePos += 1
            else:
                y = 0
                if y == int(testLabels[iterator]):
                    trueNeg += 1
                else:
                    falseNeg += 1
            iterator += 1

        # print("New Info")
        # print(trueNeg)
        # print(truePos)
        # print(falsePos)
        # print(falseNeg)
        if truePos == 0:
            truePos = 0.0000001
        curPresicion = truePos/(truePos + falsePos)
        precisions.append(curPresicion)
        curRecall = truePos/(truePos + falseNeg)
        recalls.append(curRecall)
        curF = 2 * ((curPresicion * curRecall)/(curPresicion + curRecall))
        fScores.append((curF))
        curSensitivity = truePos / (truePos + falseNeg)
        sensitivity.append(curSensitivity)
        curSpec = trueNeg / (trueNeg + falsePos)
        curSpec = 1 - curSpec
        spec.append(curSpec)

    # print(trueNeg)
    # print(truePos)
    # print(falsePos)
    # print(falseNeg)
    # print(precisions)
    # print(recalls)
    # print(fScores)

    fig, (ax1) = plt.subplots(1)
    ax1.plot(range(40), precisions, label="Precision")
    ax1.plot(range(40), recalls, label="Recall")
    ax1.plot(range(40), fScores, label="fScore")
    ax1.set_title("Precision, Recall, and F1Score")
    ax1.set_xlabel("Theta")
    ax1.set_ylabel("Mean Value")
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 1.1)

    plt.legend()
    plt.show()

    print(spec)
    print(sensitivity)
    fig, (ax3) = plt.subplots(1)
    ax3.plot(spec, sensitivity)
    ax3.set_title("ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    plt.show()

def NewweightHeatMap():
    initWeight = np.loadtxt("initialWeights.txt", delimiter=',')
    finWeight = np.loadtxt("trainedNewWeights.txt", delimiter=',')

    imgInit = initWeight.reshape((28, 28))
    imgFin = finWeight.reshape((28, 28))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imgInit, cmap="Greys")
    ax1.set_title("Inital Weights")
    ax2.imshow(imgFin, cmap="Greys")
    ax2.set_title("Trained Weights")

    plt.show()

def NewtestChallengeSet():
    challengeDataSet = np.loadtxt("challengeDataSet.txt", delimiter=',')
    challengeLabels = np.loadtxt("challengeLabels.txt", delimiter=',')
    w = np.loadtxt("trainedNewWeights.txt", delimiter=',')
    trueNeg = 0
    truePos = 0
    falseNeg = 0
    falsePos = 0
    iterator = 0
    secInt = 0
    sVal = 0
    y = 0
    theta = 28

    precisions = []
    recalls = []
    fScores = []
    sensitivity = []
    spec = []
    for num in challengeDataSet:
        secInt = 0
        sVal = 0
        for ite in challengeDataSet[iterator]:
            sVal = sVal + (w[secInt] * ite)
            secInt = secInt + 1
        if sVal > theta:
            y = 7
            if y == int(challengeLabels[iterator]):
                truePos += 1
            else:
                falsePos += 1
        else:
            y = 9
            if y == int(challengeLabels[iterator]):
                trueNeg += 1
            else:
                falseNeg += 1
        iterator += 1

    print(f'nine identifeid as zero {trueNeg}')
    print(f'seven identifeid as one {truePos}')
    print(f'nine identifeid as one {falsePos}')
    print(f'seven identifeid as zero {falseNeg}')
