#!/usr/bin/env python
# coding=utf-8

# 1. 加载数据 data: -0.017612 	14.053064	0
from numpy import mat, shape, ones, exp, arange, array


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 1.0->w0
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

# 2 define sigmoid
def sigmoid(inX):
    return 1.0/(1+exp(-inX))


# 3 梯度计算:梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat - h
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

# 3 随机梯度上升
def stocGradAscent0(dataMatIn, classLabels):
    m,n = shape(dataMatIn)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha*error*dataMatIn[i]
    return weights

# 4 plot
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    wei = gradAscent(dataArr, labelMat)
    print type(dataArr)

    wei1 = stocGradAscent0(array(dataArr), labelMat)
    print wei
    plotBestFit(wei)
    print wei1
    plotBestFit(wei1)