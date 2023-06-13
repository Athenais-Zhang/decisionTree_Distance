import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import constant, tools
import numpy as np

class DT_center:
    dataType=None
    height = 0  # 当前树的高度
    maxDepth = 0  # 最大树深度
    curDepth = 0  # 当前节点所处深度
    node = 0  # 当前树的节点数
    maxLeafSize = 0  # 叶子节点能包含的最大容量
    meanWay = None  # 计算中心点的方式
    childTree = None  # 字典类型，用于存放子树，描述树结构，【class】：{subTree}
    childTreeNum=0
    value = None  # 叶子节点值
    meanClass = None
    distanceMeasure = None  # 距离度量

    def __init__(self, dataType='numerical',distanceMeasure='euclidean',curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000):
        self.dataType=dataType
        self.distanceMeasure=distanceMeasure
        self.maxLeafSize = maxLeafSize
        self.meanWay = meanWay
        self.curDepth = curDepth + 1
        self.maxDepth = maxDepth
        self.height = 1
        self.node = 1

    def fit(self, indeices=[], cate=None, represent=None):
        # 终止条件
        if len(indeices) == 0:
            return None
        if self.meanWay== 'MEAN' or self.meanWay== 'MODE':
            self.center=represent
        elif self.meanWay== 'MEDIAN':
            self.centerIndex = represent
            self.center = constant.get_value('gl_Xtrain')[represent] if represent != None else None
        self.value = cate
        if len(indeices) <= self.maxLeafSize:
            return self

        # step1:归类
        gl_ytrain = constant.get_value('gl_ytrain')
        y_cur = [gl_ytrain[index] for index in indeices]
        cates = set(y_cur)  # 数据集类别集合
        self.childTreeNum = len(cates)

        data = {}
        for cate in cates:
            data[cate] = []
        for index in indeices:
            data[gl_ytrain[index]].append(index)

        # step2:计算各类中心点
        represents = {} # 各类中心点
        for cate in cates:
            if self.meanWay == 'MEAN':
                represents[cate] = tools.getMean(data[cate])
            elif self.meanWay == 'MEDIAN':
                represents[cate] = tools.getMedian(data[cate])
            elif self.meanWay == 'MODE':
                represents[cate] = tools.getMode(data[cate])

        # step3:二次划分
        partitionResult = {}
        for cate in cates:
            partitionResult[cate] = []
        for index in indeices:
            minDis = np.inf
            for cate in cates:
                if self.meanWay == 'MEAN' or self.meanWay == 'MODE':
                    dis = tools.calcDistance(self.dataType,self.distanceMeasure,constant.get_value('gl_Xtrain')[index], represents[cate])
                elif self.meanWay == 'MEDIAN':
                    dis = tools.getDistance(index, represents[cate])
                if dis < minDis:
                    minDis = dis
                    minCate = cate
            partitionResult[minCate].append(index)

        # step4:递归建树
        self.childTree = {}
        for cate in cates:
            if len(partitionResult[cate]) == 0 or len(partitionResult[cate]) == len(indeices):
                self.childTree[cate] = DT_center(self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
                if self.meanWay=='MEAN' or self.meanWay=='MODE':
                    self.childTree[cate].center = represents[cate]
                elif self.meanWay=='MEDIAN':
                    self.childTree[cate].centerIndex = represents[cate]
                    self.childTree[cate].center = constant.get_value('gl_Xtrain')[represents[cate]] if represents[cate] != None else None
                self.childTree[cate].value = cate
                # self.node += self.childTree[cate].node
                # self.height = max([self.height, self.childTree[cate].height + 1])
            else:
                self.childTree[cate] = DT_center(self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
                self.childTree[cate].fit(partitionResult[cate], cate, represents[cate])
                self.node += self.childTree[cate].node
                self.height = max([self.height, self.childTree[cate].height + 1])

        return self

    def predict(self, data):
        # if self.value != None:
        #     return self.value
        # if self.childTreeNum == 1:
        #     return list(self.childTree.keys())[0]
        # minDis = np.inf
        # for cate in self.childTree.keys():
        #     dis = tools.getDistance(self.centerIndex, X_test)
        #     if dis < minDis:
        #         minDis = dis
        #         minCate = cate
        # return self.childTree[minCate].predict(X_test)
        if self.childTreeNum==0:
            return self.value
        minDis= np.inf
        for cate in self.childTree:
            dis = tools.calcDistance(self.dataType,self.distanceMeasure,data, self.childTree[cate].center)
            if dis < minDis:
                minDis = dis
                minCate = cate
        return self.childTree[minCate].predict(data)

    def predictAll(self, X_test):
        y_pred = []
        for x_test in X_test:
            y_pred.append(self.predict(x_test))
        return y_pred

    def getTreeInfo(self):
        return self.node, self.height, self.maxDepth, self.maxLeafSize, self.meanWay


    def score(self, X_test, y_test):
        y_pred = self.predictAll(X_test)
        return tools.calcAccuracy(y_pred, y_test)
