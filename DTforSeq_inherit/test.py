#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 20:02
# @Author  : Yihan Zhang
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import numpy as np
from sklearn.utils import shuffle

from DT_Seq_inherit import DT_seq
from sklearn.model_selection import KFold, StratifiedKFold
from IPython.display import Image
from sklearn import tree
import pydotplus


def test(filename: str):
    f = open(filename, 'r')
    line = f.readline()
    X = []
    y = []
    while line:
        res = line.split()
        X.append(res[1:])
        y.append(res[0])
        line = f.readline()
    f.close()

    X = np.array(X, dtype=object)
    y = np.array(y)
    scores = []
    print("%s's size: %s , types: %s" % (filename, len(X), len(set(y))))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    index = 0
    for train_index, test_index in skf.split(X, y):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = shuffle(X_train, y_train)
        t = DT_seq()
        t.fit(X_train, y_train)
        t.createGraph("test%s.dot" % (str(index)))
        index += 1
        scores.append(t.score(X_test, y_test))
    print("average acc: %s  , the detail is %s \n" % (np.average(scores), scores))
    return scores


# 单个数据集交叉验证
test("activity.txt")

# 验证代码正确性
# f = open("dataset_small/activity.txt", 'r')
# line = f.readline()
# X = []
# y = []
# while line:
#     res=line.split()
#     X.append(res[1:])
#     y.append(res[0])
#     line = f.readline()
# f.close()
#
# tree=DT_seq()
# tree.fit(X,y)
# print(tree.predict(X[0]))
