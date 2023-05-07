import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from DTforVec_numerical_03_1exhaustion import predeal, constant, tools
from DTforVec_numerical_03_1exhaustion.DT_num_distance import DT_num_distance
from DTforVec_numerical_03_1exhaustion.DT_num_gini import DT_num_gini


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def contrastExperiment_numerical(self,X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = {'myTree': [], 'T_median':[],'standard': [], 'nearestCentroid': []}
        X_normalized = predeal.normalization(X)
        for train_index, test_index in skf.split(X_normalized, y):
            X_train_normalized, X_test_normalized = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

            constant.set_value('gl_Xtrain', X_train_normalized)
            constant.set_value('gl_ytrain', y_train)
            constant.set_value('gl_distances', tools.calcDistancesMetric(X_train_normalized))

            T_gini = DT_num_gini()
            indeices = [index for index in range(len(X_train_normalized))]
            T_gini.fit(indeices)
            acc['myTree'].append(T_gini.score(X_test_normalized, y_test))

            T_median=DT_num_distance(curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            indeices = [index for index in range(len(X_train_normalized))]
            T_median.fit(indeices=indeices)
            acc['T_median'].append(T_median.score(X_test_normalized, y_test))

            standardTree = DecisionTreeClassifier()
            standardTree.fit(X_train_normalized, y_train)
            acc['standard'].append(standardTree.score(X_test_normalized, y_test))

            ncd = NearestCentroid()
            ncd.fit(X_train_normalized, y_train)
            acc['nearestCentroid'].append(ncd.score(X_test_normalized, y_test))

        return acc

    def test_iris(self,curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000):
        print('iris')
        constant._init()
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_Cervical_Cancer_Behavior_Risk(self,curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000):
        print('Cervical_Cancer_Behavior_Risk')
        constant._init()
        data = pd.read_csv("../dataset/numerical/sobar-72.csv")
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_wine(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('wine')
        constant._init()
        from sklearn.datasets import load_wine
        wine = load_wine()
        X = wine.data
        y = wine.target
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_heart_deasese(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('heart_deasese')
        constant._init()
        data = pd.read_csv("../dataset/numerical/heart.csv")
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_breast_cancer(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('breast_cancer')
        constant._init()
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_Maternal_Health_Risk(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('Maternal_Health_Risk')
        constant._init()
        data = pd.read_csv("../dataset/numerical/MaternalHealthRisk.csv")
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_banknote_authentication(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('banknote_authentication')
        constant._init()
        data = pd.read_csv("../dataset/numerical/data_banknote_authentication.data", header=None)
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_Wireless_Indoor_Localization(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('Wireless_Indoor_Localization')
        constant._init()
        f = open(r"../dataset/numerical/wifi_localization.txt")
        line = f.readline()
        data_list = []
        while line:
            num = list(map(float, line.split()))
            data_list.append(num)
            line = f.readline()
        f.close()
        dataArray = np.array(data_list)
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))

    def test_Wine_Quality(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('Wine_Quality')
        constant._init()
        data = pd.read_csv("../dataset/numerical/winequality-white.csv", sep=';')
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-T_median: ', sum(acc['T_median']) / len(acc['T_median']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))





if __name__ == '__main__':
    unittest.main()
