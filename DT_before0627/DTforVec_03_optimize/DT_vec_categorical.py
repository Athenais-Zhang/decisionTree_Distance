"""
作者：张依涵
日期：2023年04月27日
时间：10：06
描述：
"""
import copy

import numpy as np
from tqdm.auto import tqdm

from decisionTree_Distance.DTforVec_03_optimize import constant, tools


class DT_vec_categorical:
    height = 0  # 树高
    maxDepth = 0  # 最大深度
    curDepth = 0  # 当前深度
    curNode = 0  # 当前节点
    maxLeafSize = 0  # 最大叶子节点数
    meanWay = None  # 平均值计算方式
    childTree = {}  # 子树
    childTreeNum = 0  # 子树数目
    value = None  # 节点值
    meanClass = None  # 平均类
    center = None  # 中心点
    def __init__(self, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        self.maxLeafSize = maxLeafSize
        self.meanWay = meanWay
        self.maxDepth = maxDepth
        self.curDepth = curDepth + 1
        self.height = 1
        self.node = 1


    def getDistance(self, index1, index2):
        gl_distances = constant.get_value('gl_distances')
        i = min(index1,index2)
        j = max(index1,index2)
        if i == j:
            dis = 0
        else:
            dis = gl_distances[i + ((j - 1) * j >> 1)]
        return dis

    @staticmethod
    def calcDistance(data1, data2):
        dis = 0
        for chIndex in range(len(data1)):
            if data1[chIndex] != data2[chIndex]:
                dis += 1
        return dis

    @staticmethod
    def calcDistancesMetric(X):
        length = len(X)
        distancesLen = (length * (length - 1)) >> 1
        distances = np.zeros(distancesLen, dtype=float)
        for j in tqdm(range(length)):
            for i in range(j):
                distances[i + ((j - 1) * j >> 1)] = DT_vec_categorical.calcDistance(X[i], X[j])
        return distances

    def fit(self, indeices=[], cate=None, represent=None):
        # 终止条件
        if len(indeices) == 0:
            return None
        self.centerIndex = represent
        self.center = constant.get_value('gl_Xtrain')[represent] if represent != None else None
        self.value = cate
        if len(indeices) <= self.maxLeafSize:
            return self
        # X = [global_var.get_value('gl_Xtrain')[index] for index in indeices]

        # 开始建树
        # step1:归类
        # step2:初始化中心点
        # step3:计算初始分支结果并暂存
        # step4:迭代所有点为中心
        # step5:递归建树

        # step1:归类
        gl_ytrain = constant.get_value('gl_ytrain')
        gl_Xtrain = constant.get_value('gl_Xtrain')
        y_cur = [gl_ytrain[index] for index in indeices]
        cates = set(y_cur)  # 数据集类别集合
        self.childTreeNum = len(cates)

        data = {}
        for cate in cates:
            data[cate] = []
        for index in indeices:
            data[gl_ytrain[index]].append(index)

        # step2:初始化中心点
        curRepresents = {}
        for cate in cates:
            curRepresents[cate] = data[cate][0]
        represents = copy.deepcopy(curRepresents)

        # step3:计算初始分支结果并暂存
        sum_gini, giniIndeies, partitionResult,dataTo = self.__calcFirstBranches(cates, represents, indeices)

        # step4:迭代所有点为中心
        for cate in tqdm(cates):
            for index in data[cate]:
                curRepresents[cate] = index
                sum_gini_temp, giniIndeies_temp, partitionResult_temp,dataTo_temp = self.__calcBranches(curRepresents,
                                                                                            indeices,partitionResult,dataTo,giniIndeies)
                if sum_gini_temp < sum_gini:
                    sum_gini=copy.deepcopy(sum_gini_temp)
                    giniIndeies=copy.deepcopy(giniIndeies_temp)
                    partitionResult=copy.deepcopy(partitionResult_temp)
                    represents=copy.deepcopy(curRepresents)
                    dataTo=copy.deepcopy(dataTo_temp)

        # step5:递归建树
        self.childTree = {}
        for cate in cates:
            if giniIndeies[cate] == 0:
                self.childTree[cate] = DT_vec_categorical(self.curDepth, self.maxLeafSize, self.meanWay,
                                              self.maxDepth)

                self.childTree[cate].centerIndex = represents[cate]
                self.childTree[cate].center = constant.get_value('gl_Xtrain')[represents[cate]] if represents[
                                                                                                       cate] != None else None
                self.childTree[cate].value = cate

                continue
            if len(partitionResult[cate]) == 0:
                self.childTree[cate] = None
                continue
            elif len(partitionResult[cate]) == len(indeices):
                self.childTree[cate] = DT_vec_categorical(self.curDepth, self.maxLeafSize, self.meanWay,
                                              self.maxDepth)
                self.childTree[cate].fit(partitionResult[cate]['indeices'].keys(), cate, represents[cate])
                continue
            self.childTree[cate] = DT_vec_categorical(self.curDepth, self.maxLeafSize, self.meanWay,
                                          self.maxDepth)
            self.childTree[cate].fit(partitionResult[cate]['indeices'].keys(), cate, represents[cate])

        return self

    def __calcFirstBranches(self, cates, represents, indeices):
        partitionResult = {}
        for cate in cates:
            partitionResult[cate] = {'isChanged': False, 'indeices': {}, 'represent': represents[cate]}
        dataTo = {}
        for index in indeices:
            dataTo[index] = {'dis': np.inf, 'cate': None}
        # step3:重新划分数据到各分支
        for index in indeices:
            minDis = np.inf
            for cate in cates:
                dis = self.getDistance(index, represents[cate])
                if dis < minDis:
                    minDis = dis
                    minCate = cate
            partitionResult[minCate]['indeices'][index] = minDis
            dataTo[index]['dis'] = minDis
            dataTo[index]['cate'] = minCate

        # step4:计算giniIndex
        giniIndeies = {}
        sum_gini = .0
        for cate in cates:
            giniIndeies[cate] = tools.getGiniIndex(partitionResult[cate]['indeices'])
            sum_gini += giniIndeies[cate] * len(partitionResult[cate]['indeices']) / len(indeices)
        return sum_gini, giniIndeies, partitionResult, dataTo

    def __calcBranches(self,  represents, indeices, prePartitionResult, preDataTo, preGiiniIndeies):
        newPartitionResult = copy.deepcopy(prePartitionResult)
        newDataTo = copy.deepcopy(preDataTo)
        newGiiniIndeies = copy.deepcopy(preGiiniIndeies)

        for index in indeices:
            preCategory = newDataTo[index]['cate']
            minDis = np.inf
            minRepresent = None
            for represent in represents:
                newDis = self.getDistance(index, represents[represent])
                if newDis < minDis:
                    minDis = newDis
                    minRepresent = represent
            if (minRepresent is not None) and (minRepresent != preCategory):
                newDataTo[index]['dis'] = minDis
                newDataTo[index]['cate'] = minRepresent
                newPartitionResult[preCategory]['indeices'].pop(index)
                newPartitionResult[preCategory]['isChanged'] = True
                newPartitionResult[minRepresent]['indeices'][index] = minDis
                newPartitionResult[minRepresent]['isChanged'] = True
            elif minRepresent is None:
                print("minRepresent is None")
                pass
            elif minRepresent == preCategory:
                newPartitionResult[preCategory]['indeices'][index] = minDis



        # step4:重新计算giniIndex
        sum_gini = .0
        for prePartitionRes in newPartitionResult:
            if newPartitionResult[prePartitionRes]['isChanged']:
                newGiiniIndeies[prePartitionRes] = tools.getGiniIndex(newPartitionResult[prePartitionRes]['indeices'])
            sum_gini += newGiiniIndeies[prePartitionRes] * len(newPartitionResult[prePartitionRes]['indeices']) / len(indeices)
        return sum_gini, newGiiniIndeies, newPartitionResult, newDataTo


    def predict(self, data):
        if self.childTreeNum==0:
            return self.value
        minDis= np.inf
        for cate in self.childTree:
            dis = self.calcDistance(data, self.childTree[cate].center)
            if dis < minDis:
                minDis = dis
                minCate = cate
        return self.childTree[minCate].predict(data)

    def predictAll(self, data):
        result = []
        for i in range(len(data)):
            result.append(self.predict(data[i]))
        return result

    def score(self, X, y):
        y_pred = self.predictAll(X)
        return tools.calcAccuracy(y, y_pred)

    def printTree(self):
        print(self.centerIndex)
        print(self.center)
        print(self.value)
        print(self.childTreeNum)
        print(self.childTree)
        for cate in self.childTree:
            self.childTree[cate].printTree()
