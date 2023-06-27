import random
from collections import Counter

import numpy as np

from DTforCurrency_2split import tools, constant


class DT_currency_2_split:
    dataType=None
    height = 0  # 树高
    maxDepth = 0  # 最大深度
    curDepth = 0  # 当前深度
    curNode = 0  # 当前节点
    maxLeafSize = 0  # 最大叶子节点数
    meanWay = None  # 平均值计算方式
    value = None  # 节点值
    distanceMeasure = None  # 距离度量
    l_child=None
    r_child=None
    dis=0.0
    represent=None
    file=None
    def __init__(self, file='',dataType='numerical',distanceMeasure='euclidean',curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        self.file=file
        self.dataType=dataType
        self.distanceMeasure=distanceMeasure
        self.maxLeafSize = maxLeafSize
        self.meanWay = meanWay
        self.maxDepth = maxDepth
        self.curDepth = curDepth + 1
        self.height = 1
        self.node = 1

    def __calcBranches(self, represent_0, indeices):
        length=len(indeices)
        gl_distances_2d = constant.get_value('gl_distances_'+self.file)
        distances=[gl_distances_2d[represent_0][x] for x in indeices]
        sorted_distances=sorted(distances, key=lambda x:x[2])
        sorted_indecies=[y[0] if y[0]!=represent_0 else y[1] for y in sorted_distances]
        gini=1.0
        split_point=0
        for i in range(len(sorted_indecies)):
            gini_l=tools.getGiniIndex(self.file,sorted_indecies[0:i])
            gini_r=tools.getGiniIndex(self.file,sorted_indecies[i:-1])
            gini_cur=(i/length)*gini_l+(1-i/length)*gini_r
            if gini_cur<gini:
                gini=gini_cur
                split_point=i
        return sorted_indecies[split_point],gini,sorted_distances,split_point



    def fit(self, indeices=[]):
        # 终止条件
        if len(indeices) == 0:
            return None
        gl_y = constant.get_value("gl_ytrain_"+self.file)
        if len(indeices) <= self.maxLeafSize:
            return self

        gini=1.0
        represents=[]
        sorted_distances=[]
        maxMonteCarloNum = constant.get_value('maxMonteCarloNum')
        times = 0
        split_point_min=0
        while times<maxMonteCarloNum:
            represent_0 = indeices[int(random.random() * len(indeices))]
            represent_1, gini_cur,sorted_distances_cur,split_point = self.__calcBranches(represent_0, indeices)
            if gini_cur<gini:
                represents=[represent_0,represent_1]
                gini=gini_cur
                sorted_distances=sorted_distances_cur.copy()
                split_point_min=split_point
            times+=1

        self.represent=constant.get_value('gl_Xtrain_'+self.file)[represents[0]]
        self.dis=constant.get_value('gl_distances_'+self.file)[represents[0]][represents[1]][-1]
        self.value=Counter([gl_y[x] for x in indeices]).most_common(1)[0][0]

        # step5:递归建树
        if represents[0]==represents[1]:
            return self
        l_indecies=[x[0] if x[0]!=represents[0] else x[1] for x in sorted_distances[0:split_point_min]]
        l_indecies.sort()
        r_indecies=[x[0] if x[0]!=represents[0] else x[1] for x in sorted_distances[split_point_min:-1]]
        r_indecies.sort()
        if len(l_indecies)==0 or len(r_indecies)==0:
            return self
        self.l_child = DT_currency_2_split(self.file,self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
        self.l_child.fit(l_indecies)
        self.r_child = DT_currency_2_split(self.file,self.dataType,self.distanceMeasure,self.curDepth, self.maxLeafSize, self.meanWay, self.maxDepth)
        self.r_child.fit(r_indecies)

        return self

    def predict(self, data):
        if self.l_child==None and self.r_child==None:
            return self.value
        if self.l_child==None and self.r_child!=None:
            return self.r_child.predict(data)
        if self.l_child!=None and self.r_child==None:
            return self.l_child.predict(data)
        dis=tools.calcDistance(self.dataType,self.distanceMeasure,data,self.represent)
        if dis>self.dis:
            return self.r_child.predict(data)
        else:
            return self.l_child.predict(data)

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


