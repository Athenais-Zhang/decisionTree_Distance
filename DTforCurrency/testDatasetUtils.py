import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from DTforCurrency.testUtils import testNumericalUtils
from DTforCurrency.testUtils import testCategoricalUtils
from DTforCurrency.testUtils import testSequenceUtils
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

def print_data_info(dataType,fileName, X,y):
    pass

class testNumericalDatasetUtils:
    def test_appendicitis(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='appendicitis', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)

    def test_bands(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='bands', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)

    def test_banknote_authentication(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        constant._init()
        data = pd.read_csv("../dataset/numerical/data_banknote_authentication.data", header=None)
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print_data_info(dataType='numerical',fileName='banknote_authentication', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        
    def test_breast_cancer(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        constant._init()
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        print_data_info(dataType='numerical',fileName='breast_cancer', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)

    def test_ecoli(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='ecoli', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)

    def test_glass(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='glass', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_haberman( curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='haberman', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_ionosphere(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='ionosphere', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_iris(curDepth=0, maxLeafSize=1, meanWay='MEDIAN', maxDepth=1000000000):
        constant._init()
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        print_data_info(dataType='numerical',fileName='iris', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_movement_libras(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='movement_libras', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_newthyroid( curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='newthyroid', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_page_block(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='page-blocks', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_penbased(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='penbased', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_pima(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='pima', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_ring(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='ring', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_satimage(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='satimage', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_segment(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='segment', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_sonar(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='sonar', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_spambase(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='spambase', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_texture(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='texture', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_twonorm(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='twonorm', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_wdbc(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='wdbc', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_wine(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        constant._init()
        from sklearn.datasets import load_wine
        wine = load_wine()
        X = wine.data
        y = wine.target
        print_data_info(dataType='numerical',fileName='wine', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_Wine_Quality_white(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        constant._init()
        data = pd.read_csv("../dataset/numerical/winequality-white.csv", sep=';')
        dataArray = data.values
        X = dataArray[:, :-1]
        y = dataArray[:, -1]
        print_data_info(dataType='numerical',fileName='winequality-white', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)


    def test_Wine_Quality_red(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
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
        print_data_info(dataType='numerical',fileName='winequality-red', X=X, y=y)
        return testNumericalUtils.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)

class testCategoricalDatasetUtils:

    def test_assistant_evaluation(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName='../dataset/categorical/Datasets/assistant-evaluation.csv'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_balance_scale(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/balance-scale.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    @staticmethod
    def test_breast_cancer_wisconsin(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/breast-cancer-wisconsin.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_car(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/car.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_chess(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/chess.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_credit_approval(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/credit-approval.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_dermatology(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/dermatology.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_dna_promoter(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/dna-promoter.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_hayes_roth(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/hayes-roth.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_heart_disease(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/heart-disease.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_house_votes(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/house-votes.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lecturer_evaluation(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/lecturer_evaluation.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lenses(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/lenses.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lung_cancer(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/lung_cancer.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_lymphography(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/lymphography.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_mammographic_mass(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/mammographic_mass.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_mushroom( curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        # fileName = 'dataset/mushroom.csv'
        fileName='../dataset/categorical/agaricus-lepiota(mushroom).data'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_nursery(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/nursery.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_photo_evaluation(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/photo_evaluation.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_primary_tumor(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/primary_tumor.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_solar_flare(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/solar_flare.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_soybean_small(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/soybean_small.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_tic_tac_toe(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/tic_tac_toe.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)

    def test_titanic(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/titanic.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)
        
    def test_zoo(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        fileName = '../dataset/categorical/Datasets/zoo.txt'
        return testCategoricalUtils.datasetTest(fileName,curDepth, maxLeafSize, meanWay, maxDepth)
    
class testSequenceDatasetUtils:

    def test_aslbu():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/aslbu.txt")

    def test_auslan2():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/auslan2.txt")

    def test_context():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/context.txt")

    def test_epitope():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/epitope.txt")

    def test_gene():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/gene.txt")

    def test_pioneer():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/pioneer.txt")

    def test_question():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/question.txt")

    def test_reuters():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/reuters.txt")

    def test_robot():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/robot.txt")

    def test_skating():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/skating.txt")

    def test_unix():
        constant._init()
        return testSequenceUtils.test_sequence_contrast("../dataset/sequence/unix.txt")