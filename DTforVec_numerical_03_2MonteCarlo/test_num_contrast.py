# python3 -m pytest -s test_num_contrast.py::MyTestCase::test_iris
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from DTforVec_numerical_03_2MonteCarlo import predeal, constant, tools
from DTforVec_numerical_03_2MonteCarlo.DT_num_distance import DT_num_distance
from DTforVec_numerical_03_2MonteCarlo.DT_num_gini import DT_num_gini
from DTforVec_numerical_03_2MonteCarlo.DT_num_gini_monteCarlo import DT_num_gini_monteCarlo


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def contrastExperiment_numerical(self,X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = {'myTree': [], 'myTree_mc':[],'T_median':[],'T_mean':[],'standard': [], 'nearestCentroid': []}
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


            T_gini_mc = DT_num_gini_monteCarlo()
            indeices = [index for index in range(len(X_train_normalized))]
            T_gini_mc.fit(indeices)
            acc['myTree_mc'].append(T_gini_mc.score(X_test_normalized, y_test))

            T_median=DT_num_distance(curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000)
            indeices = [index for index in range(len(X_train_normalized))]
            T_median.fit(indeices=indeices)
            acc['T_median'].append(T_median.score(X_test_normalized, y_test))

            T_mean = DT_num_distance(curDepth=0, maxLeafSize=1, meanWay='MEAN', maxDepth=1000000000)
            indeices = [index for index in range(len(X_train_normalized))]
            T_mean.fit(indeices=indeices)
            acc['T_mean'].append(T_mean.score(X_test_normalized, y_test))

            standardTree = DecisionTreeClassifier()
            standardTree.fit(X_train_normalized, y_train)
            acc['standard'].append(standardTree.score(X_test_normalized, y_test))

            ncd = NearestCentroid()
            ncd.fit(X_train_normalized, y_train)
            acc['nearestCentroid'].append(ncd.score(X_test_normalized, y_test))

        return acc

    def printResult(self,acc):
        print(acc)
        # print('average-myTree: ', np.mean(acc['myTree']))
        # print('average-T_mc: ', np.mean(acc['myTree_mc']))
        # print('average-T_median: ', np.mean(acc['T_median']))
        # print('average-standard: ', np.mean(acc['standard']))
        # print('average-nearestCentroid: ', np.mean(acc['nearestCentroid']))
        print('average-myTree: %.2f , the detail is %s ' %(np.mean(acc['myTree']),acc['myTree']))
        print('average-myTree_mc: %.2f , the detail is %s ' % (np.mean(acc['myTree_mc']), acc['myTree_mc']))
        print('average-T_median: %.2f , the detail is %s ' % (np.mean(acc['T_median']), acc['T_median']))
        print('average-T_mean: %.2f , the detail is %s ' % (np.mean(acc['T_mean']), acc['T_mean']))
        print('average-standard: %.2f , the detail is %s ' % (np.mean(acc['standard']), acc['standard']))
        print('average-nearestCentroid: %.2f , the detail is %s ' % (np.mean(acc['nearestCentroid']), acc['nearestCentroid']))


    def test_appendicitis(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('appendicitis')
        constant._init()
        f=open(r'../dataset/numerical/dataset/appendicitis.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values.astype(float)
        X = data[:, :-1]
        y = data[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_bands(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('bands')
        constant._init()
        f=open(r'../dataset/numerical/dataset/bands.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        data = predeal.dealMissingValue(data,'?')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_banknote_authentication(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('banknote_authentication')
        constant._init()
        data = pd.read_csv("../dataset/numerical/data_banknote_authentication.data", header=None)
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_breast_cancer(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('breast_cancer')
        constant._init()
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_ecoli(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('ecoli')
        constant._init()
        f=open(r'../dataset/numerical/dataset/ecoli.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_glass(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('glass')
        constant._init()
        f=open(r'../dataset/numerical/dataset/glass.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_haberman(self, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('haberman')
        constant._init()
        f = open(r'../dataset/numerical/dataset/haberman.dat', encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0] == '@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train = pd.DataFrame(sentimentlist)
        data = df_train.values
        X = data[:, :-1].astype(int)
        y = data[:, -1]
        class_label = LabelEncoder()
        # for index in range(len(y)):
        #     y[index] = class_label.fit_transform(y[index])
        y = class_label.fit_transform(y)
        print(X.shape)
        acc = self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_ionosphere(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('ionosphere')
        constant._init()
        f=open(r'../dataset/numerical/dataset/ionosphere.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_iris(self,curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000):
        print('iris')
        constant._init()
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_movement_libras(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('movement_libras')
        constant._init()
        f=open(r'../dataset/numerical/dataset/movement_libras.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_newthyroid(self, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('newthyroid')
        constant._init()
        f = open(r'../dataset/numerical/dataset/newthyroid.dat', encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0] == '@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train = pd.DataFrame(sentimentlist)
        data = df_train.values
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc = self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_page_block(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('page_block')
        constant._init()
        f=open(r'../dataset/numerical/dataset/page-blocks.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_penbased(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('penbased')
        constant._init()
        f=open(r'../dataset/numerical/dataset/penbased.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_pima(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('pima')
        constant._init()
        f=open(r'../dataset/numerical/dataset/pima.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_ring(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('ring')
        constant._init()
        f=open(r'../dataset/numerical/dataset/ring.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_satimage(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('satimage')
        constant._init()
        f=open(r'../dataset/numerical/dataset/satimage.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_segment(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('segment')
        constant._init()
        f=open(r'../dataset/numerical/dataset/segment.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # class_label = LabelEncoder()
        # y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_sonar(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('sonar')
        constant._init()
        f=open(r'../dataset/numerical/dataset/sonar.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_spambase(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('spambase')
        constant._init()
        f=open(r'../dataset/numerical/dataset/spambase.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_texture(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('texture')
        constant._init()
        f=open(r'../dataset/numerical/dataset/texture.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_twonorm(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('twonorm')
        constant._init()
        f=open(r'../dataset/numerical/dataset/twonorm.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_wdbc(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('wdbc')
        constant._init()
        f=open(r'../dataset/numerical/dataset/wdbc.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values
        # data = predeal.dealMissingValue(data,'0')
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        class_label = LabelEncoder()
        y=class_label.fit_transform(y)
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_wine(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('wine')
        constant._init()
        from sklearn.datasets import load_wine
        wine = load_wine()
        X = wine.data
        y = wine.target
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_Wine_Quality_white(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('Wine_Quality-white')
        constant._init()
        data = pd.read_csv("../dataset/numerical/winequality-white.csv", sep=';')
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)

    def test_Wine_Quality_red(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print('Wine_Quality-red')
        constant._init()
        f=open(r'../dataset/numerical/dataset/winequality-red.dat',encoding='utf-8')
        sentimentlist = []
        for line in f:
            if line.strip()[0]=='@':
                continue
            s = line.strip().split(',')
            sentimentlist.append(s)
        f.close()
        df_train=pd.DataFrame(sentimentlist)
        data = df_train.values.astype(float)
        X = data[:, :-1]
        y = data[:, -1]
        print(X.shape)
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        self.printResult(acc)



if __name__ == '__main__':
    unittest.main()
