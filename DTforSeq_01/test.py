import numpy as np
from sklearn.utils import shuffle

from DT_sequences import DT_seq
from sklearn.model_selection import KFold, StratifiedKFold


def test(filename: str):
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
    scores = []
    print("%s's size: %s , types: %s" % (filename, len(X), len(set(y))))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    for train_index, test_index in skf.split(X, y):
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = shuffle(X_train, y_train)
        tree = DT_seq()
        tree.fit(X_train, y_train)
        scores.append(tree.score(X_test, y_test))
    print("average acc: %s  , the detail is %s \n" % (np.average(scores), scores))
    return scores


# 所有数据集
fileNames = [
    'activity.txt',
    'aslbu.txt',
    'auslan2.txt',
    'context.txt',
    'epitope.txt',
    'gene.txt',
    'news.txt',
    'pioneer.txt',
    'question.txt',
    'reuters.txt',
    'robot.txt',
    'skating.txt',
    'unix.txt',
    'webkb.txt'
]
fileFolderNames = ["dataset", "dataset_small"]
accs = {}
# for fileName in fileNames:
#     try:
#         res = test(fileFolderNames[1] + "/" + fileName)
#         accs[fileName] = res
#     except Exception as e:
#         print("there's an error with %s, the error type is %s, detail: %s" % (fileName, type(e), e))
# print("===============================================endend===============================================")
# print(accs)

# 单个数据集交叉验证
test("dataset/activity.txt")

# 验证代码正确性
# f = open("dataset_small/activity.txt", 'r')
# line = f.readline()
# X = []
# y = []
# while line:
#     res=line.split()
#     X.append(res[1:])
#     y.append(res[0])
#     line = f.readline()
# f.close()
#
# tree=DT_seq()
# tree.fit(X,y)
# print(tree.predict(X[0]))
