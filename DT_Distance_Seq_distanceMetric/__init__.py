#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 14:27
# @Author  : Yihan Zhang
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm


"""
invoke method:
python3 -m pytest -s test.py::TestTree::test_codeCorrectly --filePathName=../dataset/pioneer.txt --distance_measure=2
python3 -m pytest -s test.py::TestTree::test_singleDataset --filePathName=../dataset/pioneer.txt --distance_measure=2
python3 -m pytest -s test.py::TestTree::test_allDatasets --distance_measure=2

"""