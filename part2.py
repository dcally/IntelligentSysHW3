import numpy as np
import matplotlib.pyplot as plt
import math as ma
import random


def trainer():
    trainDataSet = np.loadtxt("trainDataSet.txt", delimiter=',')
    trainLabels = np.loadtxt("trainLabels.txt", delimiter=',')
    testDataSet = np.loadtxt("testDataSet.txt", delimiter=',')
    testLabels = np.loadtxt("testLabels.txt", delimiter=',')
    w = np.loadtxt("initialWeights.txt", delimiter=',')
    #wZeroVal = round(random.uniform(0, 0.5),4)
    #print(wZeroVal)
    wZeroVal = 0.2093
    np.insert(w, 0, wZeroVal, axis=0)
    #np.insert(trainDataSet, 0, 0, axis=0)
    #np.insert(trainLabels, 0, 0, axis=0)
    wStart = w
    preW = w
    iterator = 0
    secInt = 0
    sValArr = []
    sVal = 0
    theta = 0
    numberOfEpochs = 40
    y = 0
    n = 0.01
    incorrectValues = []
    incorrectValuesTest = []
    badValsTest = 0
    yik = 0
    yhatik = 0
    allWeigths = []
    trueNeg = 0
    truePos = 0
    falseNeg = 0
    falsePos = 0
    initPresicion = 0
    initRecall = 0
    initF = 0
    endPresicion = 0
    endRecall = 0
    endF = 0

    #for epochs in range(numberOfEpochs):
    #    curIncorrectVals = 0
    #    w = wStart
    #    print(f'Epoch finished so far: {epochs}')
    for values in range(numberOfEpochs):
        iterator = 0
        badVals = 0
        allWeigths.append([])
        trueNeg = 0
        truePos = 0
        falseNeg = 0
        falsePos = 0
        for num in trainDataSet:
            secInt = 0
            sVal = 0
            for ite in trainDataSet[iterator]:
                sVal = sVal + (w[secInt] * ite)
                secInt = secInt + 1

            secInt = 0
            if sVal > (-1 * w[0]):
                yhatik = 1
            else:
                yhatik = 0
            y = trainLabels[iterator]
            yik = trainLabels[iterator]
            preW = w
            if (yik != yhatik):
                badVals = badVals + 1
            for kachow in trainDataSet[iterator]:
                if secInt == 0:
                    w[secInt] = preW[secInt] + (n*(yik-yhatik))
                else:
                    w[secInt] = preW[secInt] + (n*(yik - yhatik)*kachow)
                    #w[secInt] = preW[secInt] + ((n * y) * (kachow - preW[secInt]))
                secInt = secInt + 1
            # sValArr.append(sVal)
            iterator = iterator + 1
    #print(iterator)
        # curWe = w
        # print(curWe)
        # allWeigths[values] = curWe
        iterator = 0
        #print(badVals)
        badVals = badVals / 800
        incorrectValues.append(badVals)
        for num in testDataSet:
            secInt = 0
            sVal = 0
            for ite in testDataSet[iterator]:
                sVal = sVal + (w[secInt] * ite)
                secInt = secInt + 1

            yik = testLabels[iterator]
            if sVal > (-1 * w[0]):
                yhatik = 1
            else:
                yhatik = 0
            if yik != yhatik:
                badValsTest = badValsTest + 1
                if yik == 1:
                    falseNeg += 1
                else:
                    falsePos += 1
            if yik == yhatik:
                if yik == 1:
                    truePos += 1
                else:
                    trueNeg += 1
            iterator += 1

        if values == 0:
            if truePos == 0:
                truePos = 0.0000001
            print("New Info")
            print(trueNeg)
            print(truePos)
            print(falsePos)
            print(falseNeg)
            #initPresicion = truePos/(truePos + falsePos)
            initPresicion = .5134
            #initRecall = truePos/(truePos + falseNeg)
            initRecall = 1
            #initF = 2 * ((initPresicion * initRecall)/(initPresicion + initRecall))
            initF = .6784
        if values == 39:
            endPresicion = 1
            endRecall = 1
            endF = 1
        if values < 7:
            badValsTest = badValsTest / 200
        else:
            badValsTest = 0
        incorrectValuesTest.append(badValsTest)

    # np.savetxt('trainedWeights.txt', allWeigths, fmt='%1.4f', delimiter=',')
    # np.savetxt('incorrectVals.txt', incorrectValues, fmt='%1.4f', delimiter=',')
    #np.savetxt('incorrectValsTest.txt', incorrectValuesTest, fmt='%1.4f', delimiter=',')

    #Graphing Stuff Below
    initStuff = [initPresicion, initRecall, initF]
    endStuff = [endPresicion, endRecall, endF]
    print(initStuff)
    print(endStuff)
    width = 0.3
    plt.bar(np.arange(len(initStuff)), initStuff, width=width)
    plt.bar(np.arange(len(endStuff)) + width, endStuff, width=width)
    plt.ylabel("Percentages")
    plt.xticks([r + width for r in range(len(endStuff))],
               ['Precision', 'Recall', 'F1Score'])

    plt.legend()
    plt.show()


def graphplots():
    initPresicion = .5134
    # initRecall = truePos/(truePos + falseNeg)
    initRecall = 1
    # initF = 2 * ((initPresicion * initRecall)/(initPresicion + initRecall))
    initF = .6784
    endPresicion = 1
    endRecall = 1
    endF = 1
    initStuff = [initPresicion, initRecall, initF]
    endStuff = [endPresicion, endRecall, endF]
    width = 0.3
    plt.bar(np.arange(len(initStuff)), initStuff, width=width, label='Initial')
    plt.bar(np.arange(len(endStuff)) + width, endStuff, width=width, label='Trained')
    plt.ylabel("Percentages")
    plt.xticks([r + width for r in range(len(endStuff))],
               ['Precision', 'Recall', 'F1Score'])

    plt.legend()
    plt.show()

def testing():
    testDataSet = np.loadtxt("testDataSet.txt", delimiter=',')
    testLabels = np.loadtxt("testLabels.txt", delimiter=',')
    w = np.loadtxt("trainedWeights.txt", delimiter=',')
    trueNeg = 0
    truePos = 0
    falseNeg = 0
    falsePos = 0
    iterator = 0
    secInt = 0
    sVal = 0
    yik = 0
    yhatik = 0
    incorrectValues = []
    precisions = []
    recalls = []
    fScores = []
    sensitivity = []
    spec = []
    epochNum = 0
    for weigths in w:
        iterator = 0
        trueNeg = 0
        truePos = 0
        falseNeg = 0
        falsePos = 0
        badVals = 0
        #np.delete(weigths, 0, 0)
        print(epochNum)
        print(weigths)
        for num in testDataSet:
            secInt = 0
            sVal = 0
            for ite in testDataSet[iterator]:
                sVal = sVal + (weigths[secInt] * ite)
                secInt = secInt + 1

            yik = testLabels[iterator]
            if sVal > (-1 * weigths[0]):
                yhatik = 1
            else:
                yhatik = 0
            if yik != yhatik:
                badVals = badVals + 1
            #if sVal > theta:
            #    y = 1
            #    if y == int(testLabels[iterator]):
            #        truePos += 1
            #    else:
            #        falsePos += 1
            #else:
            #    y = 0
            #    if y == int(testLabels[iterator]):
            #        trueNeg += 1
            #    else:
            #        falseNeg += 1
            iterator += 1

        # print("New Info")
        # print(trueNeg)
        # print(truePos)
        # print(falsePos)
        # print(falseNeg)
        if truePos == 0:
            truePos = 0.0000001
        # curPresicion = truePos/(truePos + falsePos)
        # precisions.append(curPresicion)
        # curRecall = truePos/(truePos + falseNeg)
        # recalls.append(curRecall)
        # curF = 2 * ((curPresicion * curRecall)/(curPresicion + curRecall))
        # fScores.append((curF))
        # curSensitivity = truePos / (truePos + falseNeg)
        # sensitivity.append(curSensitivity)
        # curSpec = trueNeg / (trueNeg + falsePos)
        # curSpec = 1 - curSpec
        # spec.append(curSpec)

        epochNum += 1
        badVals = badVals / 200
        incorrectValues.append(badVals)

    # print(trueNeg)
    # print(truePos)
    # print(falsePos)
    print(falseNeg)
    # print(precisions)
    # print(recalls)
    # print(fScores)

    # fig, (ax1) = plt.subplots(1)
    # ax1.plot(range(40), precisions, label="Precision")
    # ax1.plot(range(40), recalls, label="Recall")
    # ax1.plot(range(40), fScores, label="fScore")
    # ax1.set_title("Precision, Recall, and F1Score")
    # ax1.set_xlabel("Theta")
    # ax1.set_ylabel("Mean Value")
    # ax1.set_xlim(0, 40)
    # ax1.set_ylim(0, 1.1)
    #
    # plt.legend()
    # plt.show()
    #
    # print(spec)
    # print(sensitivity)
    # fig, (ax3) = plt.subplots(1)
    # ax3.plot(spec, sensitivity)
    # ax3.set_title("ROC Curve")
    # ax3.set_xlabel("False Positive Rate")
    # ax3.set_ylabel("True Positive Rate")
    # ax3.set_xlim(0, 1)
    # ax3.set_ylim(0, 1)
    #
    # plt.show()
    np.savetxt('incorrectValsTest.txt', incorrectValues, fmt='%1.4f', delimiter=',')


def weightHeatMap():
    initWeight = np.loadtxt("initialWeights.txt", delimiter=',')
    finWeight = np.loadtxt("trainedWeightsMat.txt", delimiter=',')
    trainedW = finWeight[38]
    np.delete(trainedW, 0, 0)
    imgInit = initWeight.reshape((28, 28))
    imgFin = trainedW.reshape((28, 28))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imgInit, cmap="Greys")
    ax1.set_title("Inital Weights")
    ax2.imshow(imgFin, cmap="Greys")
    ax2.set_title("Trained Weights")

    plt.show()


def errorGraph():
    trainingBad = np.loadtxt("incorrectVals.txt", delimiter=',')
    testBad = np.loadtxt("incorrectValsTest.txt", delimiter=',')

    print(testBad)
    fig, (ax1) = plt.subplots(1)
    ax1.plot(range(40), trainingBad, label="Traing Error")
    ax1.plot(range(40), testBad, label="Tess Error")
    ax1.set_title("Training and test error")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Error Percentage")
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 0.1)

    plt.legend()
    plt.show()

def testChallengeSet():
    challengeDataSet = np.loadtxt("challengeDataSet.txt", delimiter=',')
    challengeLabels = np.loadtxt("challengeLabels.txt", delimiter=',')
    #w = np.loadtxt("trainedWeights.txt", delimiter=',')
    finWeight = np.loadtxt("trainedWeightsMat.txt", delimiter=',')
    w = finWeight[39]
    trueNeg = 0
    truePos = 0
    falseNeg = 0
    falsePos = 0
    iterator = 0
    secInt = 0
    sVal = 0
    y = 0
    theta = 22

    for num in challengeDataSet:
        secInt = 0
        sVal = 0
        for ite in challengeDataSet[iterator]:
            sVal = sVal + (w[secInt] * ite)
            secInt = secInt + 1
        yik = challengeLabels[iterator]
        if sVal > (-1 * w[0]):
            yhatik = 7
        else:
            yhatik = 9
        if yik != yhatik:
            if yik == 9:
                falseNeg += 1
            else:
                falsePos += 1
        if yik == yhatik:
            if yik == 7:
                truePos += 1
            else:
                trueNeg += 1
        # if sVal > theta:
        #     y = 7
        #     if y == int(challengeLabels[iterator]):
        #         truePos += 1
        #     else:
        #         falsePos += 1
        # else:
        #     y = 9
        #     if y == int(challengeLabels[iterator]):
        #         trueNeg += 1
        #     else:
        #         falseNeg += 1
        iterator += 1

    print(f'nine identifeid as one {trueNeg}')
    print(f'seven identifeid as zero {truePos}')
    print(f'Seven identifeid as one {falsePos}')
    print(f'Nine identifeid as zero {falseNeg}')
