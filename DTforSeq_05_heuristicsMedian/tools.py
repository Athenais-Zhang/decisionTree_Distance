#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 17:05
# @Author  : Yihan Zhang
# @Site    : 
# @File    : tools.py
# @Software: PyCharm
import queue
import random


def minDistance(word1, word2):
    '''if len(word1) == 0:
        return len(word2)
    elif len(word2) == 0:
        return len(word1)'''
    M = len(word1)
    N = len(word2)
    output = [[0] * (N + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                output[i][j] = 0
            elif i == 0 and j != 0:
                output[i][j] = j
            elif i != 0 and j == 0:
                output[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                output[i][j] = output[i - 1][j - 1]
            else:
                output[i][j] = min(output[i - 1][j - 1] + 1, output[i - 1][j] + 1, output[i][j - 1] + 1)
    return output, output[i][j]


def backtrackingPath(word1, word2):
    dp, nu = minDistance(word1, word2)
    m = len(dp) - 1
    n = len(dp[0]) - 1
    edit_distance = dp[m][n]
    operation = []

    while n > 0 or m > 0:
        if n and dp[m][n - 1] + 1 == dp[m][n]:
            operation.append("ins:" + word2[n - 1])
            n -= 1
            continue
        if m and dp[m - 1][n] + 1 == dp[m][n]:
            operation.append(word1[m - 1] + ":del")
            m -= 1
            continue
        if dp[m - 1][n - 1] + 1 == dp[m][n]:
            operation.append(word1[m - 1] + ":" + word2[n - 1])
            n -= 1
            m -= 1
            continue
        if dp[m - 1][n - 1] == dp[m][n]:
            operation.append(word1[m - 1])
        n -= 1
        m -= 1
    operation = operation[::-1]
    return operation, edit_distance


def convert(data, op):
    return data


def findCenter(dataSet):
    medianIndex = (random.random()) * len(dataSet)
    median = dataSet[medianIndex]
    trackPathSet = {}
    Op = queue.Queue()

    for data in dataSet:
        trackPath = backtrackingPath(data, median)
        # for o in trackPath[0]:
        #     if o in trackPathSet:
        #         trackPathSet[o] += 1
        #     else:
        #         trackPathSet[o] = 1
        for index in range(trackPath[1]):
            op_detail = (index, trackPath[0][index])
            if op_detail in trackPathSet:
                trackPathSet[op_detail] += 1
            else:
                trackPathSet[op_detail] = 1
    opArray = [None] * len(trackPathSet)
    for op_detail in trackPathSet:
        # opArray[trackPathSet[op_detail]]=op_detail
        if opArray[trackPathSet[op_detail]] is None:
            opArray[trackPathSet[op_detail]] = []
        opArray[trackPathSet[op_detail]].append(op_detail)

    while len(Op) > 0:
        operate = Op.get()
        median = convert(median, operate)
