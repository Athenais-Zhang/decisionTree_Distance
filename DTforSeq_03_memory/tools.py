#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/23 10:27
# @Author  : Yihan Zhang
# @Site    : 
# @File    : tools.py
# @Software: PyCharm
import math

from textdistance import LCSSeq
import numpy as np


def calDis(seq1, seq2):
    simlarity = LCSSeq().similarity(seq1, seq2)
    dis = (1 / simlarity) if simlarity != 0 else 1
    return dis


def calcDistances(X):
    length = len(X)
    distancesLen = (length * (length - 1)) >> 1
    distances = np.zeros(distancesLen, dtype=float)
    for j in range(length):
        for i in range(j):
            distances[i + ((j - 1) * j >> 1)] = calDis(X[i], X[j])
    return distances


def findCenter(XIndeices, distances):
    minDis = math.inf
    center = None
    for index1 in XIndeices:
        dis = 0
        for index2 in XIndeices:
            i = min(index1, index2)
            j = max(index1, index2)
            if i==j:
                continue
            dis += distances[i + ((j - 1) * j >> 1)]
        if minDis > dis:
            minDis = dis
            center = index1
    return center
