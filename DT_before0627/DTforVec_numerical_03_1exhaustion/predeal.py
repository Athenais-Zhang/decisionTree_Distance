"""
作者：张依涵
日期：2023年05月06日
时间：21：40
描述：
"""
import numpy as np


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