import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from DT_before0627.DTforVec_categorical_02_2MonteCarlo import constant, predeal, tools
from DT_before0627.DTforVec_categorical_02_2MonteCarlo.DT_vec_gini import DT_vec_gini


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def myTree_categoricalTest_contrast(self,X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc_gini = []
        acc_trad=[]
        acc_std=[]
        acc_ncd=[]
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, y_train = shuffle(X_train, y_train)

            constant.set_value('gl_Xtrain', X_train)
            constant.set_value('gl_ytrain', y_train)
            constant.set_value('gl_distances', tools.calcDistancesMetric(X_train))

            T_gini = DT_vec_gini()
            indeices = [index for index in range(len(X_train))]
            T_gini.fit(indeices)
            acc_gini.append(T_gini.score(X_test, y_test))

            # T_trad = DT_vec_distance(curDepth, maxLeafSize, meanWay, maxDepth)
            # T_trad.fit(X_train, y_train)
            # acc_trad.append(T_trad.score(X_test, y_test))
            #
            # standardTree = DecisionTreeClassifier()
            # standardTree.fit(X_train, y_train)
            # acc_std.append(standardTree.score(X_test, y_test))
            #
            # ncd = NearestCentroid()
            # ncd.fit(X_train, y_train)
            # acc_ncd.append(ncd.score(X_test, y_test))

        print("acc_gin: %.2f  , the detail is %s " % (np.mean(acc_gini), acc_gini))
        # print("acc_tra: %.2f  , the detail is %s " % (np.mean(acc_trad), acc_trad))
        # print("acc_std: %.2f  , the detail is %s " % (np.mean(acc_std), acc_std))
        # print("acc_ncd: %.2f  , the detail is %s " % (np.mean(acc_ncd), acc_ncd))
        # return acc

    def datasetTest(self,fileName,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print(fileName)
        constant._init()
        df = pd.read_csv(fileName, header=None)
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        print(X.shape)
        print(len(set(y)))
        self.myTree_categoricalTest_contrast(X,y)


    def test_assistant_evaluation(self, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        # fileName = 'dataset/mushroom.csv'
        fileName= '../../dataset/categorical/Datasets/assistant-evaluation.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)



    def test_balance_scale(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/balance-scale.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_breast_cancer_wisconsin(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/breast-cancer-wisconsin.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)



    def test_car(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/car.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_chess(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/chess.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_credit_approval(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/credit-approval.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_dermatology(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/dermatology.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_dna_promoter(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/dna-promoter.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_hayes_roth(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/hayes-roth.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_heart_disease(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/heart-disease.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)


    def test_house_votes(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/house-votes.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lecturer_evaluation(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/lecturer_evaluation.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lenses(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/lenses.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lung_cancer(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/lung_cancer.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lymphography(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/lymphography.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_mammographic_mass(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/mammographic_mass.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_mushroom(self, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        # fileName = 'dataset/mushroom.csv'
        fileName= '../../dataset/categorical/agaricus-lepiota(mushroom).data'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_nursery(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/nursery.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_photo_evaluation(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/photo_evaluation.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_primary_tumor(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/primary_tumor.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_solar_flare(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/solar_flare.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_soybean_small(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/soybean_small.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_tic_tac_toe(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/tic_tac_toe.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_titanic(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/titanic.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_zoo(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../../dataset/categorical/Datasets/zoo.txt'
        self.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)


if __name__ == '__main__':
    unittest.main()
