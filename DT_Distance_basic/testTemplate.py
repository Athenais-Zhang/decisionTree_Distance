"""
作者：张依涵
日期：2022年09月12日
时间：11：24
描述：建立测试模板，方便init文件中直接进行计算
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

import predeal as predeal
import DistanceTree_numerical as dt
import DistanceTree_categorical as dtc

def myTree_nemericalTest(X,y,curDepth=0, maxLeafSize=1, maxDepth=10000000):
    k = 10
    # kf = KFold(n_splits=k, shuffle=True, random_state=None)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
    acc = []
    TDepth = []
    TNode = []
    X_normalized = predeal.normalization(X)
    for train_index, test_index in skf.split(X_normalized, y):
        X_train_normalized, X_test_normalized = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

        T_mean = dt.DistanceTree_numerical(curDepth, maxLeafSize, maxDepth)
        T_mean.fit(X_train_normalized, y_train)
        acc.append(T_mean.score(X_test_normalized, y_test))
        TDepth.append(T_mean.height)
        TNode.append(T_mean.node)

    return acc,TDepth,TNode

def myTree_categoricalTest(X,y,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
    acc = []
    heights = []
    nodes=[]
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = shuffle(X_train, y_train)

        T_mean = dtc.DistanceTree_categorical(curDepth, maxLeafSize, meanWay, maxDepth)
        T_mean.fit(X_train, y_train)
        acc.append(T_mean.score(X_test, y_test))
        heights.append(T_mean.height)
        nodes.append(T_mean.node)
    return acc,heights,nodes

def contrastExperiment_categorical(X,y,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
    acc = {'myTree':[],'standard':[],'nearestCentroid':[]}
    heights = {'myTree':[],'standard':[]}
    nodes = {'myTree':[],'standard':[]}
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = shuffle(X_train, y_train)

        T_mean = dtc.DistanceTree_categorical(curDepth, maxLeafSize, meanWay, maxDepth)
        T_mean.fit(X_train, y_train)
        acc['myTree'].append(T_mean.score(X_test, y_test))
        heights['myTree'].append(T_mean.height)
        nodes['myTree'].append(T_mean.node)

        standardTree = DecisionTreeClassifier()
        standardTree.fit(X_train, y_train)
        heights['standard'].append(standardTree.get_depth() + 1)
        nodes['standard'].append(standardTree.tree_.node_count)
        acc['standard'].append(standardTree.score(X_test, y_test))

        ncd = NearestCentroid()
        ncd.fit(X_train, y_train)
        acc['nearestCentroid'].append(ncd.score(X_test, y_test))

    return acc, heights, nodes

def contrastExperiment_numerical(X,y,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
    acc = {'myTree':[],'standard':[],'nearestCentroid':[]}
    heights = {'myTree':[],'standard':[]}
    nodes = {'myTree':[],'standard':[]}
    X_normalized = predeal.normalization(X)
    for train_index, test_index in skf.split(X_normalized, y):
        X_train_normalized, X_test_normalized = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

        T_mean = dt.DistanceTree_numerical(curDepth, maxLeafSize, meanWay, maxDepth)
        T_mean.fit(X_train_normalized, y_train)
        acc['myTree'].append(T_mean.score(X_test_normalized, y_test))
        heights['myTree'].append(T_mean.height)
        nodes['myTree'].append(T_mean.node)

        standardTree = DecisionTreeClassifier()
        standardTree.fit(X_train_normalized, y_train)
        heights['standard'].append(standardTree.get_depth() + 1)
        nodes['standard'].append(standardTree.tree_.node_count)
        acc['standard'].append(standardTree.score(X_test_normalized, y_test))

        ncd = NearestCentroid()
        ncd.fit(X_train_normalized, y_train)
        acc['nearestCentroid'].append(ncd.score(X_test_normalized, y_test))

    return acc,heights,nodes




