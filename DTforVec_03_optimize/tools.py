"""
作者：张依涵
日期：2023年04月27日
时间：08：51
描述：
"""
from collections import Counter

import numpy as np

from decisionTree_Distance.DTforVec_03_optimize import constant


def getGiniIndex(Dataset):
    gl_Xtrain = constant.get_value('gl_Xtrain')
    gl_ytrain = constant.get_value('gl_ytrain')

    len_D_C__k = len(Dataset)

    y_cur = [gl_ytrain[index] for index in Dataset]
    cates=set(y_cur)
    gini_temp = .0
    for cate in cates:
        len_D_C__k_c = y_cur.count(cate)
        gini_temp +=  (len_D_C__k_c / len_D_C__k) ** 2
    gini = 1 - gini_temp
    return gini

def calcAccuracy(y, y_pred):
    errorTimes = 0
    for index in range(len(y)):
        errorTimes += 0 if y_pred[index] == y[index] else 1
    return 1 - errorTimes / len(y)


def normalization(dataArray):
    resArray = np.empty(dataArray.shape)
    for row in range(len(dataArray[0]) - 1):
        maxValue = max(dataArray[:, row])
        minValue = min(dataArray[:, row])
        for index in range(len(dataArray)):
            value = (dataArray[index][row] - minValue) / (maxValue - minValue)
            resArray[index, row] = value
    for index in range(len(dataArray)):
        resArray[index, -1] = dataArray[index, -1]
    return resArray


def  dealMissingValue(X,missingValue):
    # 缺失值处理
    for index in range(len(X[0])):
        if missingValue not in X[:, index]:
            continue
        mostcommon = None
        cnts = Counter(X[:, index])
        if cnts.most_common()[0][0] != missingValue:
            mostcommon = cnts.most_common()[0][0]
        else:
            mostcommon = cnts.most_common()[1][0]

        for row in range(len(X)):
            if X[row, index] == missingValue:
                X[row, index] = mostcommon
    return X