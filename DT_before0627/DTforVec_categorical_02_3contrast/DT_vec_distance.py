"""
作者：张依涵
日期：2023年04月27日
时间：20：42
描述：
"""
import copy
from collections import Counter
from itertools import takewhile

import numpy as np

class DT_vec_distance:
    height = 0  # 当前树的高度
    maxDepth = 0  # 最大树深度
    curDepth = 0  # 当前节点所处深度
    node = 0  # 当前树的节点数
    maxLeafSize = 0  # 叶子节点能包含的最大容量
    meanWay = None  # 计算中心点的方式
    childTree = None  # 字典类型，用于存放子树，描述树结构，【class】：{subTree}
    value = None  # 叶子节点值
    meanClass = None

    def __init__(self, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        self.maxLeafSize = maxLeafSize
        self.meanWay = meanWay
        self.maxDepth = maxDepth
        self.curDepth = curDepth + 1
        self.height = 1
        self.node = 1


    def __calcMedian(self,dataset):
        # # 计数dataset中每个元素出现次数
        # res=Counter(dataset)
        # return res.most_common(1)
        # res={}
        # for i in set(dataset):
        #     res[i]=dataset.count(i)
        # resultMaxTime = -1
        # result=None
        # for i in res:
        #     if res[i]>resultMaxTime:
        #         resultMaxTime=res[i]
        #         result=i
        # return result
        res={}
        ds=list()
        for data in dataset:
            # ds.append(data.toString())
            # str3 = '.'.join([str(x) for x in list])
            ds.append(','.join([str(x) for x in data]))
        for i in set(ds):
            res[i]=ds.count(i)
        resultMaxTime=-1
        result=None
        for i in res:
            if res[i]>resultMaxTime:
                resultMaxTime=res[i]
                result=i
        # return result
        return list(np.array(result.split(','),dtype=int))



    def __get_count(self, dct, n):
        data = dct.most_common()
        if (len(data) <= n):
            return list(data)
        else:
            val = data[n - 1][1]
            return list(takewhile(lambda x: x[1] >= val, data))  # 返回序列，当predicat

    def __get_MaxTime(self, list):
        res = list[0][0]
        for index in range(len(list)):
            if res > list[index][0]:
                res = list[index][0]
        return res

    def __calcMean(self, dataset):
        meanRes = []
        for index in range(len(dataset[0])):
            res = Counter(np.array(dataset)[:, index])
            meanRes.append(self.__get_MaxTime(self.__get_count(res, 1)))
        return meanRes

    def __calcMean2(self, dataset):
        meanRes = []
        for index in range(len(dataset[0])):
            res = Counter(np.array(dataset)[:, index])
            meanRes.append(res.most_common(1)[0][0])
        return meanRes

    def __calcDis(self, data1, data2):
        dis = 0
        for chIndex in range(len(data1)):
            if data1[chIndex] != data2[chIndex]:
                dis += 1
        return dis


    def __calcMeans(self, X, y, classes):
        cate = {}
        meanCla = {}
        for cl in classes:
            cate[cl] = []
        for index in range(len(y)):
            cate[y[index]].append(X[index])
        for cl in cate:
            # if self.meanWay is None:
            #     meanCla[cl] = self.__calcMean(cate[cl])
            # else:
            #     meanCla[cl] = self.__calcMean2(cate[cl])
            if self.meanWay=='MEAN':
                meanCla[cl]=self.__calcMean(cate[cl])
            elif self.meanWay=='MEDIAN':
                meanCla[cl]=self.__calcMedian(cate[cl])
        return meanCla

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

        # 已经选出每个类的中心 现在计算距离
        dataset = {}
        self.childTree={}
        for cl in classes:
            self.childTree[cl] = DT_vec_distance(self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
            dataset[cl] = {}
            dataset[cl]['X'] = list()
            dataset[cl]['y'] = list()
        for index in range(len(X)):
            minDis = float('inf')
            currentCl = None
            for cl in classes:
                dis = self.__calcDis(X[index], self.meanClass[cl])
                if dis < minDis:
                    minDis = dis
                    currentCl = cl
            dataset[currentCl]['X'].append(X[index])
            dataset[currentCl]['y'].append(y[index])

        for cl in classes:
            if len(dataset[cl]) == 0:
                self.childTree[cl] = None
            elif np.array(dataset[cl]['X']).shape[0] == X.shape[0]:
                self.childTree[cl].value = Counter(y).most_common(1)[0][0]
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

    def predict(self, data):
        if self.childTree == None or self.meanClass == None:
            return self.value
        minCl = None
        preDis = float('inf')
        for cl in self.meanClass:
            currentDis = self.__calcDis(data, self.meanClass[cl])
            if currentDis < preDis:
                preDis = currentDis
                minCl = cl
        return self.childTree[minCl].predict(data)

    def score(self, X, y):
        mistake = 0
        for index in range(len(X)):
            res = self.predict(X[index])
            mistake += 0 if res == y[index] else 1
        return ((X.shape[0] - mistake) / X.shape[0])
