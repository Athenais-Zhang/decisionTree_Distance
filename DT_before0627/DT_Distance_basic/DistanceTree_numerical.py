"""
作者：张依涵
日期：2022年09月14日
时间：20：01
描述：
"""
import math
import numpy as np


class DistanceTree_numerical:
    height = 0  # 当前树的高度
    maxDepth = 0  # 最大树深度
    curDepth = 0  # 当前节点所处深度
    node = 0  # 当前树的节点数
    maxLeafSize = 0  # 叶子节点能包含的最大容量
    meanWay = None  # 计算中心点的方式
    childTree = None  # 字典类型，用于存放子树，描述树结构，【class】：{subTree}
    value = None  # 叶子节点值
    meanClass = None

    def __euclideanDistance(self, data1, data2):
        sum = 0
        for row in range(len(data1)):
            sum += math.pow(data1[row] - data2[row], 2)
        return math.sqrt(sum)

    def __init__(self, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        self.maxLeafSize = maxLeafSize
        self.meanWay = meanWay
        self.curDepth = curDepth + 1
        self.maxDepth = maxDepth
        self.height = 1
        self.node = 1

    def __calcMeans(self, X, y, classes):
        dataset = {}
        for cl in classes:
            dataset[cl] = []  # 为每个类建立一个集合
        for index in range(len(X)):
            dataset[y[index]].append(X[index])  # 将每个数据添加到该类中/归类数据
        for cl in classes:
            dataset[cl] = np.array(dataset[cl])  # 每个类关联一个矩阵

        # 计算中心
        rows = len(X[0])
        meanClass = {}
        for cl in classes:
            me = []
            for row in range(rows):
                me.append(np.mean(dataset[cl][:, row]))
            meanClass[cl] = me
        return meanClass

    def fit(self, X, y):
        if X.shape[0] == 0:
            return None
        if len(set(y)) == 1:
            self.value = y[0]
            return self
        if X.shape[0] <= self.maxLeafSize or self.curDepth >= self.maxDepth:
            self.value = round(np.mean(y))
            return self

        classes = set(y)  # 数据集类别集合
        self.meanClass = self.__calcMeans(X, y, classes)  # 计算数据集中每类数据的中心点

        # 计算距离
        # distances结构：{【className1】：【distance1】，……}
        # 每个distance：列表形式存放数据集当中的所有数据和中心点的距离
        distances = {}
        for cl in classes:
            distance = []
            for num in X:
                distance.append(self.__euclideanDistance(num, np.array(self.meanClass[cl])))
            distances[cl] = distance

        self.childTree = {}
        dataset = {}
        for cl in classes:
            self.childTree[cl] = DistanceTree_numerical(self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
            dataset[cl] = {}
            dataset[cl]['X'] = list()
            dataset[cl]['y'] = list()
        for index in range(len(X)):
            minCl = list(classes)[0]
            for cl in classes:
                if distances[cl][index] <= distances[minCl][index]:
                    minCl = cl
            dataset[minCl]['X'].append(X[index])
            dataset[minCl]['y'].append(y[index])

        for cl in classes:
            if len(dataset[cl]) == 0:
                self.childTree[cl] = None
            elif np.array(dataset[cl]['X']).shape[0] == X.shape[0]:
                self.childTree[cl].value = round(np.mean(y))
            elif np.array(dataset[cl]['X']).shape[0] == 0:
                self.childTree[cl].value = cl
            else:
                self.childTree[cl].fit(
                    np.array(dataset[cl]['X']),
                    np.array(dataset[cl]['y']))

        preHeight = 0
        nodeNum = 0
        for cl in classes:
            preHeight = max(self.childTree[cl].height, preHeight)
            nodeNum += self.childTree[cl].node
        self.height = preHeight + 1
        self.node = nodeNum + 1
        return self

    def predit(self, data):
        if self.childTree == None or self.meanClass == None:
            return self.value
        minCl = None
        preDis = float('inf')
        for cl in self.meanClass:
            currentDis = self.__euclideanDistance(data, self.meanClass[cl])
            if currentDis <= preDis:
                preDis = currentDis
                minCl = cl
        return self.childTree[minCl].predit(data)

    def score(self, X, y):
        mistake = 0
        for index in range(len(X)):
            res = self.predit(X[index])
            mistake += 0 if res == y[index] else 1
        return ((X.shape[0] - mistake) / X.shape[0])

