# -*- coding:utf-8 -*-
from sklearn import svm
import numpy as np
train_data = []
train_label = []
test_data = []
test_label = []
def loadData():
    # 加载训练集
    # with open('F:/My_Program/Python_program/V1_RF_training/C1_feature/chair_feature_data.txt') as fileIn:
    #     for line in fileIn.readlines(): # readlines读取多行，readline每次读取一行
    #         lineArr = line.strip('['+']'+'\n').split(',') # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
    #         train_data.append(lineArr)
    #         train_label.append(0)
    with open('F:/My_Program/Python_program/V1_RF_training/train_face_feature_data.txt') as fileIn:
        for line in fileIn.readlines():  # readlines读取多行，readline每次读取一行
            lineArr = line.strip('[]'+'\n').split(',')  # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
            train_data.append(lineArr)
            train_label.append(1)
    with open('F:/My_Program/Python_program/V1_RF_training/train_motor_feature_data.txt') as fileIn:
        for line in fileIn.readlines():  # readlines读取多行，readline每次读取一行
            lineArr = line.strip('[]' + '\n').split(',')  # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
            train_data.append(lineArr)
            train_label.append(2)
    # 加载测试集
    # with open('F:/My_Program/Python_program/V1_RF_training/C1_feature/test_chair_feature_data.txt') as fileIn:
    #     for line in fileIn.readlines():  # readlines读取多行，readline每次读取一行
    #         lineArr = line.strip('[' + ']' + '\n').split(',')  # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
    #         test_data.append(lineArr)
    #         test_label.append(0)
    # with open('F:/My_Program/Python_program/V1_RF_training/C1_feature/test_sunflower_feature_data.txt') as fileIn:
    #     for line in fileIn.readlines():  # readlines读取多行，readline每次读取一行
    #         lineArr = line.strip('[]' + '\n').split(',')  # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
    #         test_data.append(lineArr)
    #         test_label.append(1)
    # with open('F:/My_Program/Python_program/V1_RF_training/C1_feature/test_windsor_chair_feature_data.txt') as fileIn:
    #     for line in fileIn.readlines():  # readlines读取多行，readline每次读取一行
    #         lineArr = line.strip('[]' + '\n').split(',')  # strip删除（开头）结尾处的换行符\n，split将字符串分割，默认分隔符为空格
    #         test_data.append(lineArr)
    #         test_label.append(2)
    # fileIn.close()
    return train_data, train_label, test_data, test_label # 100*3,1*100，将一个list转成matrix
# x=[[1,1],[0,0],[0,1]]
# y=[1,2,3]
# model = svm.SVC()
# model.fit(x,y)
# result=model.predict([0,-2])
# print(result)
trainx,trainy,testx,testy=loadData()
model = svm.SVC()
model.fit(trainx[0:200],trainy[0:200])

result = model.predict(trainx[170:200])

acc = 1.-float(np.count_nonzero(np.array(result)-np.array(trainy[170:200])))/len(trainy[70:100])
print(acc)
# print(list(result))
# print(trainy[:])

