"""
作者：张依涵
日期：2023年05月06日
时间：21：39
描述：
"""


def _init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    # 定义一个全局变量
    _global_dict[key] = value


def get_value(key):
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return _global_dict[key]
    except:
        print('vec02读取constant.py中' + key + '失败\r\n')
