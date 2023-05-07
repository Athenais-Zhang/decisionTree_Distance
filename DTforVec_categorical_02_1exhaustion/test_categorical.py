import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from DTforVec_categorical_02_1exhaustion import constant, tools, predeal
from DTforVec_categorical_02_1exhaustion.DT_vec_distance import DT_vec_distance
from DTforVec_categorical_02_1exhaustion.DT_vec_gini import DT_vec_gini


class MyTestCase(unittest.TestCase):
    def myTree_categoricalTest_gini(self,X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, y_train = shuffle(X_train, y_train)

            constant.set_value('gl_Xtrain', X_train)
            constant.set_value('gl_ytrain', y_train)
            constant.set_value('gl_distances', tools.calcDistancesMetric(X_train))

            T_mean = DT_vec_gini()
            indeices = [index for index in range(len(X_train))]
            T_mean.fit(indeices)
            acc.append(T_mean.score(X_test, y_test))
        print("acc: %.2f  , the detail is %s " % (np.mean(acc[-1]), acc))
        return acc

    def myTree_categoricalTest_traditional(self,X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = []
        heights = []
        nodes = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, y_train = shuffle(X_train, y_train)

            T_mean = DT_vec_distance(curDepth, maxLeafSize, meanWay, maxDepth)
            T_mean.fit(X_train, y_train)
            acc.append(T_mean.score(X_test, y_test))
            heights.append(T_mean.height)
            nodes.append(T_mean.node)
        print(acc)
        return acc, heights, nodes

    def myTree_categoricalTest_contrast(self,X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc_gini = []
        acc_trad=[]
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

            T_trad = DT_vec_distance(curDepth, maxLeafSize, meanWay, maxDepth)
            T_trad.fit(X_train, y_train)
            acc_trad.append(T_trad.score(X_test, y_test))

        print("acc_gini: %.2f  , the detail is %s " % (np.mean(acc_gini[-1]), acc_gini))
        print("acc_trad: %.2f  , the detail is %s " % (np.mean(acc_trad[-1]), acc_trad))
        # return acc

    def test_mushroom(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("test_mushroom")
        constant._init()
        df = pd.read_csv('../dataset/categorical/agaricus-lepiota(mushroom).data')
        # df = pd.read_csv('../dataset/categorical/mushroom-100.data')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        # return self.myTree_categoricalTest(X, y)
        self.myTree_categoricalTest_contrast(X,y)

    def test_house_votes(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("test_house_votes")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/house-votes.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_assistant_evaluation(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("test_assistant_evaluation")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/assistant-evaluation.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_balance_scale(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("balance-scale.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/balance-scale.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_breast_cancer_wisconsin(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("breast-cancer-wisconsin.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/breast-cancer-wisconsin.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_car(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("car.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/car.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_chess(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("chess.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/chess.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_credit_approval(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("credit-approval.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/credit-approval.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_dermatology(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("dermatology.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/dermatology.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_dna_promoter(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("dna-promoter.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/dna-promoter.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_hayes_roth(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("hayes-roth.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/hayes-roth.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_heart_disease(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("heart-disease.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/heart-disease.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_lecturer_evaluation(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("lecturer_evaluation.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/lecturer_evaluation.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_lenses(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("lenses.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/lenses.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_lung_cancer(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("lung_cancer.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/lung_cancer.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_lymphography(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("lymphography.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/lymphography.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_mammographic_mass(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("mammographic_mass.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/mammographic_mass.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_nursery(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("nursery.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/nursery.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_photo_evaluation(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("photo_evaluation.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/photo_evaluation.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_primary_tumor(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("primary_tumor.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/primary_tumor.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_solar_flare(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("solar_flare.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/solar_flare.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_soybean_small(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("soybean_small.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/soybean_small.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_tic_tac_toe(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("tic_tac_toe.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/tic_tac_toe.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_titanic(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("titanic.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/titanic.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)

    def test_zoo(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        print("zoo.txt")
        constant._init()
        df = pd.read_csv('../dataset/categorical/Datasets/zoo.txt')
        dataArr = df.values
        dataArr = predeal.dealMissingValue(dataArr, '?')
        class_label = LabelEncoder()
        for index in range(len(dataArr[0])):
            dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
        X = dataArr[:, 1:]
        y = dataArr[:, 0].astype('int64')
        self.myTree_categoricalTest_contrast(X,y)





if __name__ == '__main__':
    # unittest.main()
    # cast=unittest.TestCase()
    # cast.test_dermatology()
    pass