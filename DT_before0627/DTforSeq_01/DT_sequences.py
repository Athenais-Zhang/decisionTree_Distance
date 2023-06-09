#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 17:21
# @Author  : Yihan Zhang
# @Site    : 
# @File    : DT_sequences.py
# @Software: PyCharm
import math

import numpy as np


def findCenter(X, distances):
    minDis = math.inf
    center = None
    for x in X:
        dis = 0
        for z in X:
            dis += (distances[z[0]][x[0]] | distances[x[0]][z[0]])
        # dis = sum(distances[x[0]][0:x[0]]) + sum(distances[:, x[0]])
        if minDis > dis:
            minDis = dis
            center = x
    return center


def calcDistance(X):
    # todo 可以优化下三角矩阵的存储
    # return 下三角矩阵 存储x中各个节点的距离
    length = len(X)

    # 建立一个length*length的二维矩阵
    distances = np.zeros((length, length), dtype=int)

    # 计算edit distance并存入矩阵
    for i in range(length):
        for j in range(i):
            # minDistance(word1, word2)为计算edit distance的方法，用动态规划复杂度为平方级
            distances[i][j] = minDistance(word1=X[i], word2=X[j])
    return distances


def rank_distance(word1, word2):
    pass


def minDistance(word1, word2) -> int:
    n = len(word1)
    m = len(word2)
    if n * m == 0:
        return n + m

    # DP 数组
    D = [[0] * (m + 1) for _ in range(n + 1)]

    # 边界状态初始化
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j

    # 计算所有 DP 值
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)

    return D[n][m]


class DT_seq:
    root = None
    children = None
    depth = 0
    height = 0
    maxLeafSize = 1
    center = None

    def __init__(self, leafSize=maxLeafSize):
        self.maxLeafSize = leafSize
        pass

    def fit(self, X, y, type=None, represent=None):
        if len(X) == 0:
            return None
        self.center = represent
        self.root = type
        print("building...%s,%s"%(self.root,self.center))
        if len(set(y)) == 1 or len(X) <= self.maxLeafSize:
            return self
        # for index in range(1, len(X)):
        #     if X[index - 1] == X[index]:
        #         continue
        #     else:
        #         break
        # else:
        #     self.root = y[0]
        #     return self

        types = set(y)
        represents = {}
        data = {}
        distances = calcDistance(X)

        for type in types:
            data[type] = []
        for index in range(len(X)):
            data[y[index]].append([index, X[index]])
        for type in types:
            represent = findCenter(data[type], distances)
            represents[type] = represent

        childX = {}
        childY = {}
        for type in types:
            childX[type] = []
            childY[type] = []
        for index in range(len(X)):
            minDis = math.inf
            minRepresent = None
            for represent in represents:
                dis = distances[index][represents[represent][0]] | distances[represents[represent][0]][index]
                if dis < minDis:
                    minDis = dis
                    minRepresent = represent
            childX[minRepresent].append(X[index])
            childY[minRepresent].append(y[index])

        self.children = {}
        for type in types:
            if len(childX[type]) == 0:
                # self.children[type]=None
                continue
            elif len(childX[type]) == len(X):
                self.children[type] = DT_seq
                self.children[type].root = type
                self.children[type].center = represents[type]
            else:
                self.children[type] = DT_seq()
                self.children[type].fit(childX[type], childY[type], type, represents[type])
        # print("one")
        return self

    def predict(self, data):
        if self.children == None or len(self.children) == 1:
            return self.root
        minDis = math.inf
        closeType = self.root
        for type in self.children:
            dis = minDistance(str(data), str(self.children[type].center[1]))
            if minDis > dis:
                minDis = dis
                closeType = type
        # if len(self.children) == 1:
        #     return closeType
        return self.children[closeType].predict(data)

    def score(self, X, y):
        errorTimes = 0
        for index in range(len(X)):
            res = self.predict(X[index])
            errorTimes += 0 if res == y[index] else 1
        return ((len(X) - errorTimes) / len(X))
