#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 15:33
# @Author  : Yihan Zhang
# @Site    : 
# @File    : conftest.py
# @Software: PyCharm
# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cmdopt", action="store", default="type1", help="my option: type1 or type2"
    )

@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--cmdopt")