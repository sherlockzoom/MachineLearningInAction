#!/usr/bin/env python
# coding=utf-8

import numpy as np

#加载数据
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return np.mat(datArr)

# 实现基本pca算法
def pca(dataMat, topNfeat=999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[: -(topNfeat+1): -1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat*redEigVects.T) + meanVals
    return lowDDataMat, reconMat
