# coding=utf-8
from numpy import *
import operator

"""
ｋ近邻算法　１　计算当前点与数据之间的距离　２　对距离排序　３　取前ｋ个中频率最高的类
作为该点的类别
"""


def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inX, dataSet, labels, k):
	"""欧几里德距离 1,做差　２　平方　３求和　４　开平方"""
	data_set_size = dataSet.shape[0]
	diffMat = tile(inX, (data_set_size,1)) - dataSet
	sqdifMat = diffMat**2
	sqDistance = sqdifMat.sum(axis=1)
	sqDistance = sqDistance**0.5
	sortedDistanceIndices = sqDistance.argsort()
	classCount = {}
	for i in range(k): # 字典统计，排序找到ｋ个中出现最高频的类标
		voteLabel = labels[sortedDistanceIndices[i]]
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(),key=lambda x:x[1], reverse=True)
	return sortedClassCount[0][0]


def file2matrix(filename):
	"""数据预处理，读取文件数据到矩阵中 前3列数据，最后一列类标"""
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	retMat = zeros((numberOfLines, 3))
	classLabelVector = []
	idx = 0
	for line in arrayOLines:  # 一行处理数据
		line = line.strip()
		lst_feature = line.split('\t')
		retMat[idx,:]  = lst_feature[:3]
		classLabelVector.append(int(lst_feature[-1]))
		idx += 1
	return retMat, classLabelVector


def autoNorm(dataSet):
	"""数据归一化处理 newvalue = (oldvalue-min)/(max-min)"""
	minVal = dataSet.min(0)
	maxVal = dataSet.max(0)
	ranges = maxVal - minVal
	normat = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normat = dataSet - tile(minVal, (m,1))
	normat = normat/tile(ranges, (m,1))
	return normat, ranges, minVal

from sklearn.neighbors import KNeighborsClassifier

def sk_knn_test():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normat, ranges, minval = autoNorm(datingDataMat)
	m = normat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0
	### sklearn
	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(normat[numTestVecs:m, :], datingLabels[numTestVecs:m])
	###
	for i in range(numTestVecs):
		# clf_result = classify0(normat[i, :], normat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
		clf_result = clf.predict(normat[i, :])
		print "the classifier came back with : %d, the real answer is : %d" % (clf_result, datingLabels[i])
		if clf_result != datingLabels[i]:
			errorCount += 1.0
	print "the total error rate is : %f" % (errorCount / numTestVecs)

# 分类过程
def datingClassTest():
	"""使用前面的ｋ近邻分类器来测试预测效果"""
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normat, ranges , minval = autoNorm(datingDataMat)
	m = normat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0
	for i in range(numTestVecs):
		clf_result = classify0(normat[i,:], normat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print "the classifier came back with : %d, the real answer is : %d"%(clf_result, datingLabels[i])
		if clf_result!=datingLabels[i]:
			errorCount += 1.0
	print "the total error rate is : %f"%(errorCount/numTestVecs)

# 预测过程：允许用户输入特征得到预测结果
def classifyPerson():
	resultLst = ['不喜欢', '魅力一般', '极具魅力']
	percentTats = float(raw_input("玩游戏的时间百分比："))
	ffMiles = float(raw_input("每年获得的飞行里程数："))
	iceCream = float(raw_input("每周消耗的冰淇淋公斤数："))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normat, ranges, minval = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	clf_result = classify0((inArr-minval)/ranges, normat, datingLabels,3)
	print "你对这个人的感觉应该是：",resultLst[clf_result-1]


######### 手写识别数字 ############
def img2vector(filename):
	"""32*32图像转换为1*1024 list 读取图像转换称测试向量(一次处理一个手写字)"""
	retVector = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			retVector[0, 32*i+j] = int(lineStr[j])
	return retVector

from os import listdir
def handwritingClassTest():
	""" 分类器代码 """
	hwLabels = []
	trainFileList = listdir('trainingDigits')
	m = len(trainFileList)  # m个样例
	trainingMat = zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumber = int(fileStr.split('_')[0])  # {数字}_{编号}
		hwLabels.append(classNumber)
		trainingMat[i,:] = img2vector("trainingDigits/%s"%fileNameStr)
	# 下面构建测试集
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)

	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(trainingMat, hwLabels)


	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumber = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
		# clf_result = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		clf_result = clf.predict(vectorUnderTest)
		print "分类器返回的是：%d, 真实的结果是: %d "%(clf_result, classNumber)
		if clf_result!=classNumber:
			errorCount += 1
	print "\n 总的错误是:%d"%errorCount
	print "\n 总的误差率是: %f"%(errorCount/float(mTest))



if __name__=='__main__':
	# datingClassTest()
	# classifyPerson()
	handwritingClassTest()
	# sk_knn_test()
