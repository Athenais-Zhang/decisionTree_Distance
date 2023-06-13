
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np

import tools
import constan as constant
import apriori

class DT:
    dataType=None
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
    distanceMeasure = None  # 距离度量

    def __init__(self, dataType='numerical',distanceMeasure='euclidean',curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        self.dataType=dataType
        self.distanceMeasure=distanceMeasure
        self.maxLeafSize = maxLeafSize
        self.meanWay = meanWay
        self.maxDepth = maxDepth
        self.curDepth = curDepth + 1
        self.height = 1
        self.node = 1

    def calcRuleDistance(self,xData, rule):
        pass

    def __calcBranches(self, cates, curRules, indeices):
        partitionResult = {}
        for cate in cates:
            partitionResult[cate] = []
        gl_Xtrain = constant.get_value('gl_Xtrain')
        # step3:重新划分数据到各分支
        for index in indeices:
            minDis = np.inf
            # global minCate
            minCate = None
            for cate in cates:
                dis = self.calcRuleDistance(gl_Xtrain[index], curRules[cate])
                if dis < minDis:
                    minDis = dis
                    minCate = cate
            partitionResult[minCate].append(index)

        # step4:计算giniIndex
        giniIndeies = {}
        sum_gini = .0
        for cate in cates:
            giniIndeies[cate] = tools.getGiniIndex(partitionResult[cate])
            sum_gini += giniIndeies[cate] * len(partitionResult[cate]) / len(indeices)
        return sum_gini, giniIndeies, partitionResult

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

        # 找最显著的规则
        curRules={}
        for cate in cates:
            curRules[cate]=apriori.mostConfRule(gl_Xtrain[data[cate]],gl_ytrain[data[cate]])
        rules=curRules.copy()

        # # step2:初始化中心点
        # curRepresents = {}
        # for cate in cates:
        #     curRepresents[cate] = data[cate][0]
        # represents = curRepresents.copy()

        # step3:计算初始分支结果并暂存
        sum_gini, giniIndeies, partitionResult = self.__calcBranches(cates, curRules, indeices)


        # maxMonteCarloNum = constant.get_value('maxMonteCarloNum')
        # monteCarloNum = 1
        # for cate in cates:
        #     monteCarloNum *= len(data[cate])
        # monteCarloNum /= 10
        # monteCarloNum = int(min(maxMonteCarloNum, monteCarloNum))
        # times = 0
        # while times < monteCarloNum:
        #     for cate in cates:
        #         random_index = np.random.randint(0, len(data[cate]))
        #         curRepresents[cate] = data[cate][random_index]
        #     sum_gini_temp, giniIndeies_temp, partitionResult_temp = self.__calcBranches(cates, curRepresents, indeices)
        #     times += 1
        #     if sum_gini_temp < sum_gini:
        #         sum_gini = sum_gini_temp
        #         giniIndeies = giniIndeies_temp
        #         partitionResult = partitionResult_temp
        #         represents = curRepresents.copy()
        #         times = 0

        # step5:递归建树
        self.childTree={}
        for cate in cates:
            if giniIndeies[cate] == 0:
                self.childTree[cate] = DT_currency(self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)

                self.childTree[cate].centerIndex = represents[cate]
                self.childTree[cate].center = constant.get_value('gl_Xtrain')[represents[cate]] if represents[cate] != None else None
                self.childTree[cate].value = cate

                continue
            if len(partitionResult[cate]) == 0:
                self.childTree[cate] = DT_currency(self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)

                self.childTree[cate].centerIndex = represents[cate]
                self.childTree[cate].center = constant.get_value('gl_Xtrain')[represents[cate]] if represents[cate] != None else None
                self.childTree[cate].value = cate
                continue
            elif len(partitionResult[cate]) ==len(indeices):
                self.childTree[cate] = DT_currency(self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
                # self.childTree[cate].fit(partitionResult[cate], cate, represents[cate])

                self.childTree[cate].centerIndex = represents[cate]
                self.childTree[cate].center = constant.get_value('gl_Xtrain')[represents[cate]] if represents[cate] != None else None
                self.childTree[cate].value = cate
                continue
            # elif len(set(partitionResult[cate]))==1:
            #     continue
            res=tools.checkPartition(partitionResult[cate],cate)
            if res is not None:
                self.childTree[cate] = DT_currency(self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)

                self.childTree[cate].centerIndex = represents[cate]
                self.childTree[cate].center = constant.get_value('gl_Xtrain')[represents[cate]] if represents[cate] != None else None
                self.childTree[cate].value = res
                continue
            self.childTree[cate] = DT_currency(self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
            self.childTree[cate].fit(partitionResult[cate], cate, represents[cate])

        return self

    def predict(self, data):
        if self.childTreeNum==0:
            return self.value
        minDis= np.inf
        for cate in self.childTree:
            dis = tools.calcDistance(self.dataType,self.distanceMeasure,data, self.childTree[cate].center)
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


