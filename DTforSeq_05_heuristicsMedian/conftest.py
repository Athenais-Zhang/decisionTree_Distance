#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 17:03
# @Author  : Yihan Zhang
# @Site    : 
# @File    : conftest.py
# @Software: PyCharm

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--distance_measure", action="store", default="-1", help="my option: 0,1 or 2"
    )
    parser.addoption(
        "--filePathName", action="store", default="../dataset/activity.txt", help="path"
    )

@pytest.fixture
def distance_measure(request):
    return request.config.getoption("--distance_measure")


@pytest.fixture
def filePathName(request):
    return request.config.getoption("--filePathName")
