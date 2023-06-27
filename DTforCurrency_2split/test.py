import os
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from DTforCurrency_2split import constant, tools, predeal
from DTforCurrency_2split import DT_currency as dt

import sys
sys.setrecursionlimit(1000000000)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    sequenceFileNames = [
        'aslbu',
        'auslan2',
        'context',
        'epitope',
        'gene',
        'pioneer',
        'question',
        'reuters',
        'robot',
        'skating',
        'unix'
    ]
    sequenceFileFolderNames = ["../dataset/sequence/"]

    def test_sequences(self):
        constant._init()
        dataSetName='../dataset/sequence/context.txt'
        # filename=self.sequenceFileFolderNames[0]+self.sequenceFileNames[i]
        f = open(dataSetName, 'r')
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
        acc = {'myTree_mc_rank':[],'T_median_rank':[],'myTree_mc_edit':[],'T_median_edit': []}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        index = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, y_train = shuffle(X_train, y_train)
            constant.set_value('gl_Xtrain_'+dataSetName, X_train)
            constant.set_value('gl_ytrain_'+dataSetName, y_train)
            constant.set_value('gl_distances_'+dataSetName, tools.calcDistancesMetric_2d('sequence','rank',X_train))

            t_rank = dt.DT_currency_2_split(dataSetName,dataType='sequence',distanceMeasure='rank')
            indeices = [index for index in range(len(X_train))]
            t_rank.fit(indeices)
            index += 1
            acc['myTree_mc_rank'].append(t_rank.score(X_test, y_test))
        print(acc)

        accWrite = []
        # 判断acc中是否包含某个key
        if 'T_median_rank' in acc.keys() and len(acc['T_median_rank']) != 0:
            accWrite.append(format(np.mean(acc['T_median_rank']),'.3f'))


        # 如果文件中还没有数据，就写入数据，如果有数据，向后追加数据
        fileName="result_sequence.txt"
        if not os.path.exists(fileName):
            with open(fileName, 'w') as f:
                f.write('%s,'%(dataSetName))
                f.write(','.join(accWrite))
        else:
            with open(fileName, 'a+') as f:
                f.write('%s,'%(dataSetName))
                f.write(','.join(accWrite))

        #向文件中添加空行
        with open(fileName, 'a+') as f:
            f.write('\n')
        return acc

    categoricalFileNames = [
        'assistant_evaluation',
        'balance_scale',
        'breast_cancer_wisconsin',
        'car',
        'chess',
        'credit_approval',
        'dermatology',
        'dna_promoter',
        'hayes_roth',
        'heart_disease',
        'house_votes',
        'lecturer_evaluation',
        'lenses',
        'lung_cancer',
        'lymphography',
        'mammographic_mass',
        'mushroom',
        'nursery',
        'photo_evaluation',
        'primary_tumor',
        'solar_flare',
        'soybean_small',
        'tic_tac_toe',
        'titanic',
        'zoo'
    ]

    @staticmethod
    def myTree_categoricalTest_contrast(fileName, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        constant._init()
        df = pd.read_csv(fileName, header=None)
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = {'myTree_mc':[],'T_median':[],'T_mean':[],'standard': [], 'nearestCentroid': []}
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, y_train = shuffle(X_train, y_train)
            constant.set_value('gl_Xtrain_'+fileName, X_train)
            constant.set_value('gl_ytrain_'+fileName, y_train)
            constant.set_value('gl_distances_'+fileName, tools.calcDistancesMetric_2d('categorical','hanming',X_train))
            indeices = [index for index in range(len(X_train))]

            T_gini_mc = dt.DT_currency_2_split(fileName,dataType='categorical',distanceMeasure='hanming')
            T_gini_mc.fit(indeices)
            acc['myTree_mc'].append(T_gini_mc.score(X_test, y_test))

        return acc
    def test_categorical_mushroom(self):
        fileName='../dataset/categorical/agaricus-lepiota(mushroom).data'
        acc=self.myTree_categoricalTest_contrast(fileName)
        print(acc)
        return acc

    def test_categorical_house(self):
        fileName='../dataset/categorical/Datasets/house-votes.txt'
        acc=self.myTree_categoricalTest_contrast(fileName)
        print(acc['myTree_mc'])
        print(np.mean(acc['myTree_mc']))
        return acc

    def test_numerical_iris(self):
        constant._init()
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = {'myTree_mc':[]}
        X_normalized = predeal.normalization(X)
        for train_index, test_index in skf.split(X_normalized, y):
            # X_train_normalized, X_test_normalized = X[train_index], X[test_index]
            X_train_normalized, X_test_normalized = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

            constant.set_value('gl_Xtrain_'+'iris', X_train_normalized)
            constant.set_value('gl_ytrain_'+'iris', y_train)
            constant.set_value('gl_distances_'+'iris', tools.calcDistancesMetric_2d('numerical','euclidean',X_train_normalized))

            T_gini_mc = dt.DT_currency_2_split('iris')
            indeices = [index for index in range(len(X_train_normalized))]
            T_gini_mc.fit(indeices)
            acc['myTree_mc'].append(T_gini_mc.score(X_test_normalized, y_test))
        print(acc['myTree_mc'])
        print(np.mean(acc['myTree_mc']))
        return acc


if __name__ == '__main__':
    unittest.main()
