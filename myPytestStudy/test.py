#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 14:02
# @Author  : Yihan Zhang
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import pytest

from myPytestStudy import constant


class TestDemo():
  def test_case1(self):
    constant.data["aaa"] = 1
    print("test_1")

  def test_case2(self):
    print(constant.data["aaa"])
    print("test_2")

  def test(self,cmdopt):
    print()
    print(cmdopt)
    print("aaaaaaaaa")


# def test1():
#     constant.data["aaa"] = 1
#
# def test2():
#     print(constant.data["aaa"])
