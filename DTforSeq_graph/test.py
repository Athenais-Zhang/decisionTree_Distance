#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 20:02
# @Author  : Yihan Zhang
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import datetime
import numpy as np
from sklearn.utils import shuffle
from DT_Seq_inherit import DT_seq
from sklearn.model_selection import KFold, StratifiedKFold


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
    starttime = datetime.datetime.now()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    index = 0
    for train_index, test_index in skf.split(X, y):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = shuffle(X_train, y_train)
        t = DT_seq()
        t.fit(X_train, y_train)
        t.createGraph("fileDot/test_%s_%s.dot" % (filename.split("/")[-1].split(".")[0], str(index)))
        index += 1
        scores.append(t.score(X_test, y_test))
    print("average acc: %.2f  , the detail is %s " % (np.average(scores), scores))
    endtime = datetime.datetime.now()
    print("5-fold run time：" + str((endtime - starttime).seconds) + "s\n")
    return scores


def test_codeCorrectly(filePathName):
    # 验证代码正确性
    f = open(filePathName, 'r')
    line = f.readline()
    X = []
    y = []
    while line:
        res = line.split()
        X.append(res[1:])
        y.append(res[0])
        line = f.readline()
    f.close()

    tree = DT_seq()
    tree.fit(X, y)
    print(tree.predict(X[0]))


# 所有数据集
fileNames = [
    'activity.txt',
    'aslbu.txt',
    'auslan2.txt',
    'context.txt',
    'epitope.txt',
    'gene.txt',
    'news.txt',
    'pioneer.txt',
    'question.txt',
    'reuters.txt',
    'robot.txt',
    'skating.txt',
    'unix.txt',
    'webkb.txt'
]
fileFolderNames = ["dataset"]


def test_allDataset():
    accs = {}
    for fileName in fileNames:
        try:
            res = test(fileFolderNames[0] + "/" + fileName)
            accs[fileName] = res
        except Exception as e:
            print("there's an error with %s, the error type is %s, detail: %s\n" % (fileName, type(e), e))
    print("===============================================endend===============================================")


if __name__ == '__main__':
    # test_codeCorrectly()
    # test("./dataset/activity.txt")
    test_allDataset()
