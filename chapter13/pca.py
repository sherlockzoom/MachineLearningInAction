#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#加载数据
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]   #使用高阶函数 map
    return np.mat(datArr)

# 实现基本pca算法
def pca(dataMat, topNfeat=999999):
    meanVals = np.mean(dataMat, axis=0) #求均值
    meanRemoved = dataMat - meanVals    #去中心化
    covMat = np.cov(meanRemoved, rowvar=0)  #协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))   #计算特征值 特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[: -(topNfeat+1): -1]
    redEigVects = eigVects[:, eigValInd]    #取前N个最大特征值
    lowDDataMat = meanRemoved*redEigVects   #
    reconMat = (lowDDataMat*redEigVects.T) + meanVals
    return lowDDataMat, reconMat


# 可视化pca结果
def drawPca(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.savefig('./pca.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    import pca
    dataMat = pca.loadDataSet('testSet.txt')
    lowDMat, reconMat = pca.pca(dataMat, 1)
    print lowDMat.shape
    drawPca(dataMat, reconMat)

