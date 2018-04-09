# -*- coding:utf-8 -*-

import brian2 as b2
from brian2 import *
import numpy as np
import matplotlib.cm as cmap  # 权值可视化画图的参数,colormap
import matplotlib.pyplot as plt
import time  # 记录程序运行时间
import os.path  # 图片路径处理的库
from scipy.misc import imread, imresize
import cv2
# import cPickle as pickle  # 对python对象序列化的保存和恢复
# import brian_no_units  # import it to deactivate unit checking --> This should NOT be done for testing/debugging
# import brian as b
from struct import unpack  # 解压缩 将C语言格式数据转换成Python格式数据
from glob import glob  # 处理图像路径的库
prefs.codegen.target = 'weave' # Brian takes the Python code and strings in your model and generates code in one of
# several possible different languages and actually executes that
# matplotlib.use('Agg')