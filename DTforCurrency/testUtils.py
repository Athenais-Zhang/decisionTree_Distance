import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

import predeal
import constant
import tools
import DT_gini_mtcl as dt
import DT_center as dtc

IF_Print=False

def print_data_info(dataType,fileName, X,y):
    pass

def printResult(acc):
    if IF_Print==False:
        return
    print(acc)
    if len(acc['myTree_mc']) != 0:
        print('average-myTree_mc: %.3f , the detail is %s ' % (np.mean(acc['myTree_mc']), acc['myTree_mc']))
    if len(acc['T_median']) != 0:
        print('average-T_median: %.3f , the detail is %s ' % (np.mean(acc['T_median']), acc['T_median']))
    if len(acc['T_mean']) != 0:
        print('average-T_mean: %.3f , the detail is %s ' % (np.mean(acc['T_mean']), acc['T_mean']))
    if len(acc['standard']) != 0:
        print('average-standard: %.3f , the detail is %s ' % (np.mean(acc['standard']), acc['standard']))
    if len(acc['nearestCentroid']) != 0:
        print('average-nearestCentroid: %.3f , the detail is %s ' % (np.mean(acc['nearestCentroid']), acc['nearestCentroid']))

class testNumericalUtils:
    @staticmethod
    def contrastExperiment_numerical(X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = {'myTree_mc':[],'T_median':[],'T_mean':[],'standard': [], 'nearestCentroid': []}
        X_normalized = predeal.normalization(X)
        for train_index, test_index in skf.split(X_normalized, y):
            X_train_normalized, X_test_normalized = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

            constant.set_value('gl_Xtrain', X_train_normalized)
            constant.set_value('gl_ytrain', y_train)
            constant.set_value('gl_distances', tools.calcDistancesMetric('numerical','euclidean',X_train_normalized))

            T_gini_mc = dt.DT_currency()
            indeices = [index for index in range(len(X_train_normalized))]
            T_gini_mc.fit(indeices)
            acc['myTree_mc'].append(T_gini_mc.score(X_test_normalized, y_test))

            T_median=dtc.DT_center(dataType='numerical',distanceMeasure='euclidean',curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            indeices = [index for index in range(len(X_train_normalized))]
            T_median.fit(indeices=indeices)
            acc['T_median'].append(T_median.score(X_test_normalized, y_test))

            T_mean = dtc.DT_center(dataType='numerical',distanceMeasure='euclidean',curDepth=0, maxLeafSize=1, meanWay='MEAN', maxDepth=1000000000)
            indeices = [index for index in range(len(X_train_normalized))]
            T_mean.fit(indeices=indeices)
            acc['T_mean'].append(T_mean.score(X_test_normalized, y_test))

            standardTree = DecisionTreeClassifier()
            standardTree.fit(X_train_normalized, y_train)
            acc['standard'].append(standardTree.score(X_test_normalized, y_test))

            ncd = NearestCentroid()
            ncd.fit(X_train_normalized, y_train)
            acc['nearestCentroid'].append(ncd.score(X_test_normalized, y_test))
        printResult(acc)
        return acc

class testCategoricalUtils:
    @staticmethod 
    def myTree_categoricalTest_contrast(X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = {'myTree_mc':[],'T_median':[],'T_mean':[],'standard': [], 'nearestCentroid': []}
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, y_train = shuffle(X_train, y_train)
            constant.set_value('gl_Xtrain', X_train)
            constant.set_value('gl_ytrain', y_train)
            constant.set_value('gl_distances', tools.calcDistancesMetric('categorical','hanming',X_train))
            indeices = [index for index in range(len(X_train))]

            T_gini_mc = dt.DT_currency(dataType='categorical',distanceMeasure='hanming')
            T_gini_mc.fit(indeices)
            acc['myTree_mc'].append(T_gini_mc.score(X_test, y_test))

            T_median = dtc.DT_center('categorical',distanceMeasure='hanming',curDepth=100000, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            T_median.fit(indeices)
            acc['T_median'].append(T_median.score(X_test, y_test))

            T_mean = dtc.DT_center('categorical',distanceMeasure='hanming',curDepth=100000, maxLeafSize=1, meanWay='MODE', maxDepth=1000000000)
            T_mean.fit(indeices)
            acc['T_mean'].append(T_mean.score(X_test, y_test))

            standardTree = DecisionTreeClassifier()
            standardTree.fit(X_train, y_train)
            acc['standard'].append(standardTree.score(X_test, y_test))

            ncd = NearestCentroid()
            ncd.fit(X_train, y_train)
            acc['nearestCentroid'].append(ncd.score(X_test, y_test))
        return acc
    
    @staticmethod
    def datasetTest(fileName,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        constant._init()
        df = pd.read_csv(fileName, header=None)
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        print_data_info('categorical',fileName, X,y)
        acc=testCategoricalUtils.myTree_categoricalTest_contrast(X,y)
        printResult(acc)
        return acc
    
class testSequenceUtils:
    @staticmethod
    def test_sequence_contrast(filename: str):
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
        acc = {'myTree_mc_rank':[],'T_median_rank':[],'myTree_mc_lcs':[],'T_median_lcs': [],'myTree_mc_edit':[],'T_median_edit': []}
        print_data_info('sequence',filename, X,y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        index = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, y_train = shuffle(X_train, y_train)
            constant.set_value('gl_Xtrain', X_train)
            constant.set_value('gl_ytrain', y_train)

            # 利用rank距离
            # constant.set_value('gl_distances', tools.calcDistancesMetric('sequence','rank',X_train))

            # t_rank = dt.DT_currency(dataType='sequence',distanceMeasure='rank')
            # indeices = [index for index in range(len(X_train))]
            # t_rank.fit(indeices)
            # index += 1
            # acc['myTree_mc_rank'].append(t_rank.score(X_test, y_test))


            # T_median_rank = dtc.DT_center('sequence',distanceMeasure='rank',curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            # T_median_rank.fit(indeices)
            # acc['T_median_rank'].append(T_median_rank.score(X_test, y_test))


            # 利用lcs距离
            # constant.set_value('gl_distances', tools.calcDistancesMetric('sequence','lcstr',X_train))

            # t_lcs = dt.DT_currency(dataType='sequence',distanceMeasure='lcstr')
            # indeices = [index for index in range(len(X_train))]
            # t_lcs.fit(indeices)
            # index += 1
            # acc['myTree_mc_lcs'].append(t_lcs.score(X_test, y_test))

            # T_median_lcs = dtc.DT_center('sequence',distanceMeasure='lcstr',curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            # T_median_lcs.fit(indeices)
            # acc['T_median_lcs'].append(T_median_lcs.score(X_test, y_test))


            # # 利用edit距离 
            # constant.set_value('gl_distances', tools.calcDistancesMetric('sequence','edit',X_train))

            # t_edit = dt.DT_currency(dataType='sequence',distanceMeasure='edit')
            # indeices = [index for index in range(len(X_train))]
            # t_edit.fit(indeices)
            # index += 1
            # acc['myTree_mc_edit'].append(t_edit.score(X_test, y_test))


            # T_median_edit = dtc.DT_center('sequence',distanceMeasure='edit',curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            # T_median_edit.fit(indeices)
            # acc['T_median_edit'].append(T_median_edit.score(X_test, y_test))


            # pip install python-Levenshtein
            distanceMeasure='Levenshtein'
            constant.set_value('gl_distances', tools.calcDistancesMetric('sequence',distanceMeasure,X_train))

            t_edit = dt.DT_currency(dataType='sequence',distanceMeasure=distanceMeasure)
            indeices = [index for index in range(len(X_train))]
            t_edit.fit(indeices)
            index += 1
            acc['myTree_mc_edit'].append(t_edit.score(X_test, y_test))


            T_median_edit = dtc.DT_center('sequence',distanceMeasure=distanceMeasure,curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            T_median_edit.fit(indeices)
            acc['T_median_edit'].append(T_median_edit.score(X_test, y_test))

        printResult(acc)
        return acc