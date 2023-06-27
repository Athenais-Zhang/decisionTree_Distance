"""
作者：张依涵
日期：2023年05月06日
时间：21：32
描述：
"""
import math

import numpy as np
from tqdm.auto import tqdm

from DT_before0627.DTforVec_numerical_03_1exhaustion import constant


def calcDistance(data1, data2):
    return __euclideanDistance(data1, data2)

def __euclideanDistance(data1, data2):
    sum = 0
    for row in range(len(data1)):
        sum += math.pow(data1[row] - data2[row], 2)
    return math.sqrt(sum)

def calcDistancesMetric(X):
    length = len(X)
    distancesLen = (length * (length - 1)) >> 1
    distances = np.zeros(distancesLen, dtype=float)
    for j in tqdm(range(length)):
        for i in range(j):
            distances[i + ((j - 1) * j >> 1)] = calcDistance(X[i], X[j])
    return distances

def getDistance(index1,index2):
    gl_distances = constant.get_value('gl_distances')
    i = min(index1,index2)
    j = max(index1,index2)
    if i == j:
        dis = 0
    else:
        dis = gl_distances[i + ((j - 1) * j >> 1)]
    return dis


def calcAccuracy(y, y_pred):
    errorTimes = 0
    for index in range(len(y)):
        errorTimes += 0 if y_pred[index] == y[index] else 1
    return 1 - errorTimes / len(y)


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


def checkPartition(dataset, cate):
    xSet= constant.get_value('gl_Xtrain')[dataset]
    length=len(xSet)
    for i in range(1,length):
        if ((xSet[0]==xSet[i]).all() == False):
            return None
    # return np.argmax(np.bincount(constant.get_value('gl_ytrain')[dataset]))
    return np.argmax(np.bincount(constant.get_value('gl_ytrain')[dataset].astype(int)))
    # return cate




def getMean(dataset_cate_index):
    gl_Xtrain = constant.get_value('gl_Xtrain')
    dataset_cate= [gl_Xtrain[index] for index in dataset_cate_index]
    mean=[]
    for row in range(len(dataset_cate)):
        mean.append(np.mean(dataset_cate[:, row]))
    return mean


def getMedian(dataset_cate_index):
    medianIndex=None
    dis=np.inf
    for i in dataset_cate_index:
        dis_temp=0
        for j in dataset_cate_index:
            dis_temp+=getDistance(i,j)
        if dis_temp<dis:
            dis=dis_temp
            medianIndex=i
    return medianIndex