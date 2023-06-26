from collections import Counter
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
from tqdm.auto import tqdm
import textdistance
# from textdistance import LCSSeq

import constant


def calcDistancesMetric(dataType,distanceMeasure,X):
    length = len(X)
    distancesLen = (length * (length - 1)) >> 1
    distances = np.zeros(distancesLen, dtype=float)
    for j in range(length):
        for i in range(j):
            distances[i + ((j - 1) * j >> 1)] = calcDistance(dataType,distanceMeasure,X[i], X[j])
    return distances

def calcDistance(dataType,distanceMeasure,data1, data2):
    if dataType=='numerical':
        if distanceMeasure=='euclidean':
            return __euclideanDistance(data1, data2)
    elif dataType=='categorical':
        if distanceMeasure=='hanming':
            return __hanmingDistance(data1, data2)
    elif dataType=='sequence':
        if distanceMeasure=='rank':
            return __rankDistance(data1, data2)
        elif distanceMeasure=='lcstr':
            return __distance_measure_lcstr(data1, data2)
        elif distanceMeasure=='edit':
            return __editDistance(data1, data2)
    else:
        # TODO
        return
    
def __euclideanDistance(data1, data2):
    sum = 0
    for row in range(len(data1)):
        sum += math.pow(data1[row] - data2[row], 2)
    return math.sqrt(sum)

def __hanmingDistance(data1, data2):
    dis = 0
    for chIndex in range(len(data1)):
        if data1[chIndex] != data2[chIndex]:
            dis += 1
    return dis

def __rankDistance(seq1, seq2):
    u1 = {}
    u2 = {}
    for index in range(len(seq1)):
        if seq1[index] not in u1:
            u1[seq1[index]] = []
        u1[seq1[index]].append(index + 1)

    for index in range(len(seq2)):
        if seq2[index] not in u2:
            u2[seq2[index]] = []
        u2[seq2[index]].append(index + 1)

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

def __distance_measure_lcstr(seq1, seq2):
    if type(seq1)!=list:
        seq1=seq1.tolist()
    if type(seq2)!=list:
        seq2=seq2.tolist()
    similarity = len(textdistance.lcsstr(seq1, seq2))
    distance = 1 - similarity / (len(seq1) + len(seq2))
    return distance


def getDistance(index1,index2):
    gl_distances = constant.get_value('gl_distances')
    i = min(index1,index2)
    j = max(index1,index2)
    if i == j:
        dis = 0
    else:
        dis = gl_distances[i + ((j - 1) * j >> 1)]
    return dis


def getGiniIndex(Dataset):
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
    # 如果数据集中的数据都相同，则返回该数据集的类别，其类别由出现次数最多的y决定。否则返回None
    xSet=constant.get_value('gl_Xtrain')[dataset]
    length=len(xSet)
    for i in range(1,length):
        if type(xSet[0]==xSet[i])==bool:
            if (xSet[0]==xSet[i])==False:
                return None
            else:
                continue
        elif ((xSet[0]==xSet[i]).all() == False):
            return None
    # return np.argmax(np.bincount(constant.get_value('gl_ytrain')[dataset]))
    return np.argmax(np.bincount(constant.get_value('gl_ytrain')[dataset].astype(int)))
    # return cate

    
def calcAccuracy(y, y_pred):
    errorTimes = 0
    for index in range(len(y)):
        errorTimes += 0 if y_pred[index] == y[index] else 1
    return 1 - errorTimes / len(y)



def getMean(dataset_cate_index):
    gl_Xtrain = constant.get_value('gl_Xtrain')
    dataset_cate= np.array([gl_Xtrain[index] for index in dataset_cate_index])
    mean=[]
    for row in range(len(dataset_cate[0])):
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

def getMode(dataset_cate_index):
    gl_Xtrain = constant.get_value('gl_Xtrain')
    dataset_cate= np.array([gl_Xtrain[index] for index in dataset_cate_index])
    mode=[]
    for row in range(len(dataset_cate[0])):
        res = Counter(dataset_cate[:, row])
        mode.append(res.most_common(1)[0][0])
    return mode



def __editDistance(word1, word2) -> int:
        n = len(word1)
        m = len(word2)
        
        # 有一个字符串为空串
        if n * m == 0:
            return n + m
        
        # DP 数组
        D = [ [0] * (m + 1) for _ in range(n + 1)]
        
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