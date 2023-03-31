import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

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