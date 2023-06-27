"""
作者：张依涵
日期：2022年09月14日
时间：19：58
描述：本项目主要用于修正上一版本当中的bug:
    bug描述：
        部分数据集出现dict has no distribute height错误：当某数据集都离某一中心点较远时，X.shape[0]==0会出现该错误
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import testTemplate as tt
import numpy as np

import predeal as predeal


# 结果计算numerical数据
def printNumerical(X,y,dataName,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    print('\nnumerical：',dataName)
    acc,heights,nodes = tt.contrastExperiment_numerical(X,y,curDepth, maxLeafSize, meanWay, maxDepth)
    # print(acc)
    print(
        'm',"{:.3}".format(np.mean(acc['myTree'])),
        's',"{:.3}".format(np.mean(acc['standard'])),
        'n',"{:.3}".format(np.mean(acc['nearestCentroid']))
    )
    print(
        'm',"{:.3}".format(100*np.std(acc['myTree'])),
        's',"{:.3}".format(100*np.std(acc['standard'])),
        'n',"{:.3}".format(100*np.std(acc['nearestCentroid']))
    )
    print(
        'm',np.mean(heights['myTree']),
        's',np.mean(heights['standard']),
    )
    print(
        'm',np.mean(nodes['myTree']),
        's',np.mean(nodes['standard']),
    )

# 结果计算categorical数据
def printCategorical(X,y,dataName,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    print('\ncategorical：',dataName)
    acc,heights,nodes = tt.contrastExperiment_categorical(X,y,curDepth, maxLeafSize, meanWay, maxDepth)
    print(acc)
    print(
        'm',"{:.3}".format(np.mean(acc['myTree'])),
        's',"{:.3}".format(np.mean(acc['standard'])),
        'n',"{:.3}".format(np.mean(acc['nearestCentroid']))
    )
    print(
        'm',"{:.3}".format(100*np.std(acc['myTree'])),
        's',"{:.3}".format(100*np.std(acc['standard'])),
        'n',"{:.3}".format(100*np.std(acc['nearestCentroid']))
    )
    print(
        'm',np.mean(heights['myTree']),
        's',np.mean(heights['standard']),
    )
    print(
        'm',np.mean(nodes['myTree']),
        's',np.mean(nodes['standard']),
    )


def Cervical_Cancer_Behavior_Risk(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv("../../dataset/numerical/sobar-72.csv")
    dataArray = data.values
    X = dataArray[:, :-1]
    y = dataArray[:, -1]
    printNumerical(X,y,'Cervical Cancer Behavior Risk',curDepth, maxLeafSize, meanWay, maxDepth)

def iris(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    printNumerical(X,y,'iris',curDepth, maxLeafSize, meanWay, maxDepth)

def wine(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    from sklearn.datasets import load_wine
    wine = load_wine()
    X = wine.data
    y = wine.target
    printNumerical(X,y,'wine',curDepth, maxLeafSize, meanWay, maxDepth)

def heart_deasese(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv("../../dataset/numerical/heart.csv")
    dataArray = data.values
    X = dataArray[:, :-1]
    y = dataArray[:, -1]
    printNumerical(X,y,'heart disease',curDepth, maxLeafSize, meanWay, maxDepth)

def breast_cancer(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    from sklearn.datasets import load_breast_cancer
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    printNumerical(X,y,'breast cancer',curDepth, maxLeafSize, meanWay, maxDepth)

def Maternal_Health_Risk(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv("../../dataset/numerical/MaternalHealthRisk.csv")
    dataArray = data.values
    X = dataArray[:, :-1]
    y = dataArray[:, -1]
    printNumerical(X,y,'Maternal Health Risk',curDepth, maxLeafSize, meanWay, maxDepth)

def banknote_authentication(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv("../../dataset/numerical/data_banknote_authentication.data")
    dataArray = data.values
    X = dataArray[:, :-1]
    y = dataArray[:, -1]
    printNumerical(X,y,'banknote authentication',curDepth, maxLeafSize, meanWay, maxDepth)

def Wireless_Indoor_Localization(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    f = open(r"../../dataset/numerical/wifi_localization.txt")
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    dataArray = np.array(data_list)
    X = dataArray[:, :-1]
    y = dataArray[:, -1]
    printNumerical(X,y,'Wireless Indoor Localization',curDepth, maxLeafSize, meanWay, maxDepth)

def Wine_Quality(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv("../../dataset/numerical/winequality-white.csv", sep=';')
    dataArray = data.values
    X = dataArray[:, :-1]
    y = dataArray[:, -1]
    printNumerical(X,y,'Wine Quality',curDepth, maxLeafSize, meanWay, maxDepth)

def balance_scale(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv('../../dataset/categorical/balance-scale.data')
    dataArr = data.values
    dataArr = predeal.dealMissingValue(dataArr, '?')
    class_label = LabelEncoder()
    for index in range(len(dataArr[0])):
        dataArr[:,index]=class_label.fit_transform(dataArr[:,index])
    X = dataArr[:, 1:]
    y = dataArr[:, 0].astype('int64')
    printCategorical(X,y,'balance-scale',curDepth, maxLeafSize, meanWay, maxDepth)

def tic_tac_toe(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv('../../dataset/categorical/tic-tac-toe.data')
    dataArr = data.values
    dataArr = predeal.dealMissingValue(dataArr, '?')
    class_label = LabelEncoder()
    for index in range(len(dataArr[0])):
        dataArr[:,index]=class_label.fit_transform(dataArr[:,index])
    X = dataArr[:, 1:]
    y = dataArr[:, 0].astype('int64')
    printCategorical(X,y,'tic-tac-toe',curDepth, maxLeafSize, meanWay, maxDepth)

def Car_Evaluation(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv('../../dataset/categorical/car.data')
    dataArr = data.values
    dataArr = predeal.dealMissingValue(dataArr, '?')
    class_label = LabelEncoder()
    for index in range(len(dataArr[0])):
        dataArr[:,index]=class_label.fit_transform(dataArr[:,index])
    X = dataArr[:, 1:]
    y = dataArr[:, 0].astype('int64')
    printCategorical(X,y,'Car Evaluation',curDepth, maxLeafSize, meanWay, maxDepth)

def mushroom(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    df = pd.read_csv('../../dataset/categorical/agaricus-lepiota(mushroom).data')
    dataArr = df.values
    dataArr = predeal.dealMissingValue(dataArr, '?')
    class_label = LabelEncoder()
    for index in range(len(dataArr[0])):
        dataArr[:,index]=class_label.fit_transform(dataArr[:,index])
    X = dataArr[:, 1:]
    y = dataArr[:, 0].astype('int64')
    printCategorical(X,y,'mushroom',curDepth, maxLeafSize, meanWay, maxDepth)

def house_votes_84(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    data = pd.read_csv('../../dataset/categorical/voting-records_house-votes-84.csv')
    dataArr = data.values
    dataArr = predeal.dealMissingValue(dataArr, '?')
    class_label = LabelEncoder()
    for index in range(len(dataArr[0])):
        dataArr[:,index]=class_label.fit_transform(dataArr[:,index])
    X = dataArr[:, 1:]
    y = dataArr[:, 0].astype('int64')
    printCategorical(X,y,'house-votes-84',curDepth, maxLeafSize, meanWay, maxDepth)

def breast_cancer(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    # df=pd.read_csv('../dataset/categorical/Datasets/breast-cancer-wisconsin.txt')
    file=open('../../dataset/categorical/Datasets/breast-cancer-wisconsin.txt')
    dataArr=[]
    for line in file.readlines():
        line=line.strip().split(',')
        dataArr.append(line)
    dataArr=np.array(dataArr)
    dataArr= predeal.dealMissingValue(dataArr, '?')
    class_label=LabelEncoder()
    for index in range(len(dataArr[0])):
        dataArr[:,index]=class_label.fit_transform(dataArr[:,index])
    X=dataArr[:,1:]
    y=dataArr[:,0].astype('int64')
    printCategorical(X,y,'breast-cancer-wisconsin',curDepth, maxLeafSize, meanWay, maxDepth)


if __name__ == '__main__':
    # Cervical_Cancer_Behavior_Risk(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # iris(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # # wine(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # # heart_deasese(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=8)
    # # breast_cancer(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=8)
    # # Maternal_Health_Risk(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=8)
    # banknote_authentication(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # Wireless_Indoor_Localization(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # # Wine_Quality(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=8)
    # balance_scale(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=8)
    # tic_tac_toe(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # Car_Evaluation(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # mushroom(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    house_votes_84(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    # breast_cancer(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000)
    pass
