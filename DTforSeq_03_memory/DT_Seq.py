#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/23 08:51
# @Author  : Yihan Zhang
# @Site    : 
# @File    : DT_Seq.py
# @Software: PyCharm
import math

from DTforSeq_03_memory import global_var
from DTforSeq_03_memory.tools import findCenter, calDis


class DT_seq():
    root = None
    children = None
    maxLeafSize = 1
    centerIndex = None
    center = None
    height = 1
    nodeNum = 1

    def __init__(self, leafSize=maxLeafSize):
        self.maxLeafSize = leafSize
        # pass

    # def fit(self,X,y,length):
    #      pass
    def fit(self, indeices, cate=None, represent=None):
        if len(indeices) == 0:
            return None
        self.centerIndex = represent
        self.center = global_var.get_value('gl_Xtrain')[represent] if represent!=None else None
        self.root = cate
        if len(indeices) <= self.maxLeafSize:
            return self
        # X = [global_var.get_value('gl_Xtrain')[index] for index in indeices]
        gl_ytrain = global_var.get_value('gl_ytrain')
        y_cur = [gl_ytrain[index] for index in indeices]
        cates = set(y_cur)
        if len(cates) == 1:
            return self

        # gl_Xtrain = global_var.get_value('gl_Xrain')
        gl_distances = global_var.get_value('gl_distances')

        represents = {}
        data = {}

        for c in cates:
            data[c] = []
        for index in indeices:
            data[gl_ytrain[index]].append(index)
        for c in cates:
            represent = findCenter(data[c], gl_distances)
            represents[c] = represent

        childIndeices = {}
        for c in cates:
            childIndeices[c] = []
        for index in indeices:
            minDis = math.inf
            minRepresent = None
            for represent in represents:
                i = min(index, represents[represent])
                j = max(index, represents[represent])
                if i==j:
                    continue
                dis = gl_distances[i + ((j - 1) * j >> 1)]
                if dis < minDis:
                    minDis = dis
                    minRepresent = represent
            childIndeices[minRepresent].append(index)

        self.children = {}
        for c in cates:
            if len(childIndeices[c]) == 0:
                continue
            elif len(childIndeices[c]) == len(indeices):
                self.children[c] = DT_seq()
                self.children[c].root = c
                self.children[c].centerIndex = represents[c]
                self.children[c].center = global_var.get_value('gl_Xtrain')[represents[c]]
            else:
                self.children[c] = DT_seq()
                self.children[c].fit(childIndeices[c], c, represents[c])

        return self

    def predict(self, data):
        if self.children == None or len(self.children) == 1:
            return self.root
        minDis = math.inf
        closeType = self.root
        for c in self.children:
            dis = calDis(data,self.children[c].center)
            if minDis > dis:
                minDis = dis
                closeType = c
        return self.children[closeType].predict(data)

    def score(self, X, y):
        errorTimes = 0
        for index in range(len(X)):
            res = self.predict(X[index])
            errorTimes += 0 if res == y[index] else 1
        return ((len(X) - errorTimes) / len(X))
