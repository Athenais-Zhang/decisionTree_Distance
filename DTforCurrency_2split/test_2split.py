import threading
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
max_iter=10
class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_sequence(self,fileFolder,file):
        constant._init()
        dataSetName=fileFolder+file+'.txt'
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
            constant.set_value('gl_Xtrain_'+file, X_train)
            constant.set_value('gl_ytrain_'+file, y_train)
            constant.set_value('gl_distances_'+file, tools.calcDistancesMetric_2d('sequence','rank',X_train))

            t_rank = dt.DT_currency_2_split(file,dataType='sequence',distanceMeasure='rank')
            indeices = [index for index in range(len(X_train))]
            t_rank.fit(indeices)
            index += 1
            acc['myTree_mc_rank'].append(t_rank.score(X_test, y_test))
        print(acc['myTree_mc_rank'])
        return np.mean(acc['myTree_mc_rank'])


    def worker(self,thread_num, param1, param2):
        try:
            # 这里是每个线程要执行的任务，使用传递的参数
            results=[]
            for i in range(max_iter):
                acc = self.test_sequence(param1, param2)
                results.append(acc)
            result=np.mean(results)
            # 将结果保存到文件
            # 如果文件中还没有数据，就写入数据，如果有数据，向后追加数据
            fileName="results_sequence.txt"
            if not os.path.exists(fileName):
                with open(fileName, 'w') as f:
                    f.write('%s,'%(param2))
                    # f.write(','.join(result))
                    f.write(str(result))
            else:
                with open(fileName, 'a+') as f:
                    f.write('%s,'%(param2))
                    # f.write(','.join(result))
                    f.write(str(result))

            #向文件中添加空行
            with open(fileName, 'a+') as f:
                f.write('\n')
        except Exception as e:
            # 当出现异常时捕获并打印错误信息
            print(f'Error in Thread {thread_num}: {param2},{str(e)}')


    def execute_threads(self,num_threads, params):
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=self.worker, args=(i, params[i][0], params[i][1]))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def test_sequences(self):
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
        # 调用函数并指定线程数和参数
        # parameters = [
        #     (sequenceFileFolderNames[0],sequenceFileNames[0]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[1]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[2]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[3]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[4]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[5]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[6]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[7]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[8]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[9]),
        #     (sequenceFileFolderNames[0],sequenceFileNames[10])
        #               ]  # 每个线程的参数
        # x = len(parameters)  # 线程数量
        # self.execute_threads(x, parameters)
        for file in sequenceFileNames:
            self.worker(0,sequenceFileFolderNames[0],file)

if __name__ == '__main__':
    unittest.main()
