"""
作者：张依涵
日期：2023年05月07日
时间：11：20
描述：
"""
import numpy as np

from DTforVec_numerical_03_exhaustion import constant, tools


class DT_num_distance:
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

    def __init__(self, curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000):
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
        self.centerIndex = represent
        self.center = constant.get_value('gl_Xtrain')[represent] if represent != None else None
        self.value = cate
        if len(indeices) <= self.maxLeafSize:
            return self

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

        # step2:计算各类中心点
        represents = {} # 各类中心点
        for cate in cates:
            if self.meanWay == 'MEAN':
                represents[cate] = tools.getMean(data[cate])
            elif self.meanWay == 'MEDIAN':
                represents[cate] = tools.getMedian(data[cate])

        # step3:二次划分
        partitionResult = {}
        for cate in cates:
            partitionResult[cate] = []
        for index in indeices:
            minDis = np.inf
            for cate in cates:
                dis = tools.getDistance(index, represents[cate])
                if dis < minDis:
                    minDis = dis
                    minCate = cate
            partitionResult[minCate].append(index)

        # step4:递归建树
        self.childTree = {}
        for cate in cates:
            if len(partitionResult[cate]) == 0 or len(partitionResult[cate]) == len(indeices):
                self.childTree[cate] = DT_num_distance(self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
                self.childTree[cate].centerIndex = represents[cate]
                self.childTree[cate].center = constant.get_value('gl_Xtrain')[represents[cate]] if represents[cate] != None else None
                self.childTree[cate].value = cate
                # self.node += self.childTree[cate].node
                # self.height = max([self.height, self.childTree[cate].height + 1])
            else:
                self.childTree[cate] = DT_num_distance(self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
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
            dis = tools.calcDistance(data, self.childTree[cate].center)
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
