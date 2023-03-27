#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/26 14:34
# @Author  : Yihan Zhang
# @Site    : 
# @File    : tools.py
# @Software: PyCharm
import math

import numpy as np
import textdistance
from textdistance import LCSSeq
from tqdm.auto import tqdm


def rankDistance(s1, s2):
    u1 = {}
    u2 = {}
    for index in range(len(s1)):
        if s1[index] not in u1:
            u1[s1[index]] = []
        u1[s1[index]].append(index + 1)

    for index in range(len(s2)):
        if s2[index] not in u2:
            u2[s2[index]] = []
        u2[s2[index]].append(index + 1)

    dis = 0
    for item in u1:
        if item not in u2:
            for pos in u1[item]:
                dis += pos
        else:
            p1 = u1[item]
            p2 = u2[item]
            len1 = len(p1)
            len2 = len(p2)
            if len1 > len2:
                for index in range(len2):
                    dis += abs(p1[index] - p2[index])
                for index in range(len2, len1):
                    dis += p1[index]
            else:
                for index in range(len1):
                    dis += abs(p1[index] - p2[index])
                for index in range(len1, len2):
                    dis += p2[index]
            u2.pop(item)
    for item in u2:
        for pos in u2[item]:
            dis += pos
    return dis


def lcsSeq_similarity(s1, s2):
    # simlarity = LCSSeq().similarity(s1, s2)
    similarity = len(textdistance.lcsseq(s1, s2))
    dis = (1 / similarity) if similarity != 0 else 1
    return dis


def distance_measure_lcstr(seq1, seq2):
    similarity = len(textdistance.lcsstr(seq1, seq2))
    distance = 1 - similarity / (len(seq1) + len(seq2))
    return distance


def calDis(seq1, seq2):
    # dis = lcsSeq_similarity(seq1, seq2)
    # dis = rankDistance(seq1, seq2)
    # dis = distance_measure_lcstr(seq1, seq2)
    return dis


def calcDistances(X):
    length = len(X)
    distancesLen = (length * (length - 1)) >> 1
    distances = np.zeros(distancesLen, dtype=float)
    for j in tqdm(range(length)):
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
            if i == j:
                continue
            dis += distances[i + ((j - 1) * j >> 1)]
        if minDis > dis:
            minDis = dis
            center = index1
    return center
