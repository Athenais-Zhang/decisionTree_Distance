"""
作者：张依涵
日期：2023年04月26日
时间：21：34
描述：
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from DT_before0627.DTforVec_categorical_02_1exhaustion import constant
from DT_before0627.DTforVec_categorical_02_1exhaustion import tools, predeal
from DT_before0627.DTforVec_categorical_02_1exhaustion.DT_vec_gini import DT_vec_gini
import sys
sys.setrecursionlimit(50000)



def myTree_categoricalTest(X,y,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
    acc = []
    heights = []
    nodes=[]
    index = 0
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
        index+=1
        # heights.append(T_mean.height)
        # nodes.append(T_mean.node)
        print("acc: %.2f  , the detail is %s " % (acc[-1], acc))
    return acc



def test_hayes(curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
    print("test_assistant_evaluation")
    constant._init()
    df = pd.read_csv('../../dataset/categorical/Datasets/hayes-roth2.txt', sep=',', header=None)
    dataArr = df.values
    dataArr = predeal.dealMissingValue(dataArr, '?')
    class_label = LabelEncoder()
    for index in range(len(dataArr[0])):
        dataArr[:, index] = class_label.fit_transform(dataArr[:, index])
    X = dataArr[:, 1:]
    y = dataArr[:, 0].astype('int64')
    myTree_categoricalTest(X,y)


def test_codeCorrectly(filePathName):
    # 验证代码正确性
    # f = open(filePathName, 'r')
    # line = f.readline()
    # X = []
    # y = []
    # while line:
    #     res = line.split()
    #     X.append(res[1:])
    #     y.append(res[0])
    #     line = f.readline()
    # f.close()
    # constant._init()

    df = pd.read_csv('../../dataset/categorical/Datasets/hayes-roth.txt', sep=',', header=None)
    dataArr = df.values
    dataArr = predeal.dealMissingValue(dataArr, '?')
    # class_label = LabelEncoder()
    # for index in range(len(dataArr[0])):
    #     dataArr[:,index]=class_label.fit_transform(dataArr[:,index])
    X = dataArr[:, 1:]
    y = dataArr[:, 0].astype('int64')

    constant.set_value('gl_Xtrain', X)
    constant.set_value('gl_ytrain', y)
    constant.set_value('gl_distances', tools.calcDistancesMetric(X))
    tree = DT_vec_gini()
    indeices = [index for index in range(len(X))]
    tree.fit(indeices)
    # tree.printTree()
    print(tree.predict(X[0]))




if __name__ == '__main__':
    constant._init()
    # constant.set_value('distance_measure', distance_measures[int(sys.argv[1])])

    # mushroom()
    # preTime=time.time()
    # res = mushroom()
    # print(res)
    # print('mushroom',time.time()-preTime)
    test_codeCorrectly('../dataset/categorical/mushroom-100.data')
    # test_hayes()
    print("end")