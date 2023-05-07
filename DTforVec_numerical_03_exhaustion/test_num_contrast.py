import unittest

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from DTforVec_numerical_03_exhaustion import predeal, constant, tools
from DTforVec_numerical_03_exhaustion.DT_num_gini import DT_num_gini


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_iris(self,curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        constant._init()
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        acc=self.contrastExperiment_numerical(X, y, curDepth, maxLeafSize, meanWay, maxDepth)
        print(acc)
        print('average-myTree: ', sum(acc['myTree']) / len(acc['myTree']))
        print('average-standard: ', sum(acc['standard']) / len(acc['standard']))
        print('average-nearestCentroid: ', sum(acc['nearestCentroid']) / len(acc['nearestCentroid']))


    def contrastExperiment_numerical(self,X, y, curDepth=0, maxLeafSize=1, meanWay=None, maxDepth=1000000000):
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        acc = {'myTree': [], 'standard': [], 'nearestCentroid': []}
        X_normalized = predeal.normalization(X)
        for train_index, test_index in skf.split(X_normalized, y):
            X_train_normalized, X_test_normalized = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

            constant.set_value('gl_Xtrain', X_train_normalized)
            constant.set_value('gl_ytrain', y_train)
            constant.set_value('gl_distances', tools.calcDistancesMetric(X_train_normalized))

            T_mean = DT_num_gini()
            indeices = [index for index in range(len(X_train_normalized))]
            T_mean.fit(indeices)
            acc['myTree'].append(T_mean.score(X_test_normalized, y_test))

            standardTree = DecisionTreeClassifier()
            standardTree.fit(X_train_normalized, y_train)
            acc['standard'].append(standardTree.score(X_test_normalized, y_test))

            ncd = NearestCentroid()
            ncd.fit(X_train_normalized, y_train)
            acc['nearestCentroid'].append(ncd.score(X_test_normalized, y_test))

        return acc



if __name__ == '__main__':
    unittest.main()
