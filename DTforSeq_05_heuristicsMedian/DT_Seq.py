#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 17:04
# @Author  : Yihan Zhang
# @Site    : 
# @File    : DT_Seq.py
# @Software: PyCharm
from DTforSeq_05_heuristicsMedian import constant
from DTforSeq_05_heuristicsMedian.tools import findCenter


class DT_seq():
    root = None
    children = None
    maxLeafSize = 1
    centerIndex = None
    center = None
    height = 1
    nodeNum = 1
    length = 0

    def __init__(self, leafSize=maxLeafSize):
        self.maxLeafSize = leafSize
        # pass

    def fit(self, indeices, cate=None, represent=None):
        if len(indeices)==0:
            return None
        self.centerIndex=represent
        self.center = constant.get_value('gl_Xtrain')[represent] if represent != None else None
        self.root = cate
        self.length = len(indeices)
        if len(indeices) <= self.maxLeafSize:
            return self
        gl_ytrain=constant.get_value('gl_ytrain')
        y_cur = [gl_ytrain[index] for index in indeices]
        cates = set(y_cur)
        if len(cates) == 1:
            return self

        represents = {}
        data = {}

        for c in cates:
            data[c] = []
        for index in indeices:
            data[gl_ytrain[index]].append(index)
        for c in cates:
            represent = findCenter(data[c])
            represents[c] = represent

    def predict(self, param):
        pass

    def score(self, X_test, y_test):
        pass