# -*- coding:utf-8 -*-
'''
Created on 2017.8.7

@author: Jiaxing Liu
'''
# 主要功能：创建每个卷积层的初始权值矩阵
import scipy.ndimage as sp
import numpy as np
import pylab
from parameters import *


def randomDelay(minDelay, maxDelay):  # 产生一个随机延迟
    return np.random.rand() * (maxDelay - minDelay) + minDelay


def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j * (2 * np.pi / size) * cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2 * np.pi)) / (2 * np.pi)
    return cur_pos


def sparsenMatrix(baseMatrix, pConn):  # 将一个权值矩阵变为稀疏的矩阵
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0] * int(numTargetWeights)
    while numWeights < int(numTargetWeights):
        idx = (np.int32(np.random.rand() * baseMatrix.shape[0]), np.int32(np.random.rand() * baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList


def create_weights():
    LGN_inputnum = S1_cov_patch**2
    nE = 15 * 15 # 第一层网络大小
    nI = nE
    dataPath = './random/'
    weight = {}
    weight['ee_input'] = 0.65  # 0.3
    weight['ei_input'] = 0.2
    # weight['ee'] = 0.1
    # weight['ei'] = 10.5  # 兴奋到抑制连接的权值
    # # weight['ie'] = 17.0 # 抑制连接的权值
    # weight['ie'] = 12.5
    # weight['ii'] = 0.4
    pConn = {}
    pConn['ee_input'] = 1.0
    # pConn['ei_input'] = 0.1
    # pConn['ee'] = 1.0
    # pConn['ei'] = 0.0025
    # pConn['ie'] = 0.9
    # pConn['ii'] = 0.1

    print 'create random connection matrices from input->S1'
    connNameList = ['XeAe', 'XeBe', 'XeCe', 'XeDe']
    for name in connNameList:
        randlist = 1.2 * np.random.random(LGN_inputnum) + 0.05
        weightMatrix = (np.ones((nE, LGN_inputnum)) * randlist).transpose()
        # weightMatrix = np.random.random((nInput, nE)) + 0.01
        weightMatrix *= weight['ee_input']
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            weightList = [(i, j, weightMatrix[i, j]) for j in range(nE) for i in range(LGN_inputnum)]
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)

    C1_inputnum = S2_cov_patch**2
    print 'create random connection matrices from C1->S2'
    connNameList = ['C1AeS2Ae', 'C1BeS2Ae', 'C1CeS2Ae', 'C1DeS2Ae', 'C1AeS2Be', 'C1BeS2Be', 'C1CeS2Be', 'C1DeS2Be',
                     'C1AeS2Ce', 'C1BeS2Ce', 'C1CeS2Ce', 'C1DeS2Ce', 'C1AeS2De', 'C1BeS2De', 'C1CeS2De', 'C1DeS2De',
                     'C1AeS2Ee', 'C1BeS2Ee', 'C1CeS2Ee', 'C1DeS2Ee', 'C1AeS2Fe', 'C1BeS2Fe', 'C1CeS2Fe', 'C1DeS2Fe',
                     'C1AeS2Ge', 'C1BeS2Ge', 'C1CeS2Ge', 'C1DeS2Ge', 'C1AeS2He', 'C1BeS2He', 'C1CeS2He', 'C1DeS2He']
    for name in connNameList:
        randlist = 0.85 * np.random.random(C1_inputnum) + 0.05
        weightMatrix = (np.ones((S2_num, C1_inputnum)) * randlist).transpose()
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            weightList = [(i, j, weightMatrix[i, j]) for j in range(S2_num) for i in range(C1_inputnum)]
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)

    # print 'create connection matrices from E->I which are purely random'
    # connNameList = ['XeAi']
    # for name in connNameList:
    #     weightMatrix = np.random.random((nInput, nI))
    #     weightMatrix *= weight['ei_input']
    #     weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei_input'])
    #     print 'save connection matrix', name
    #     np.save(dataPath + name, weightList)
    #
    # print 'create connection matrices from E->I which are purely random'
    # connNameList = ['AeAi', 'BeBi', 'CeCi', 'DeDi']
    # for name in connNameList:
    #     if nE == nI:
    #         weightList = [(i, i, weight['ei']) for i in range(nE)]
    #     else:
    #         weightMatrix = np.random.random((nE, nI))
    #         weightMatrix *= weight['ei']
    #         weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
    #     print 'save connection matrix', name
    #     np.save(dataPath + name, weightList)
    #
    # print 'create connection matrices from I->E which are purely random'
    # connNameList = ['AiBe', 'AiCe', 'AiDe', 'BiAe', 'BiCe', 'BiDe', 'CiAe', 'CiBe', 'CiDe', 'DiAe', 'DiBe', 'DiCe']
    # for name in connNameList:
    #     if nE == nI:
    #         weightList = [(i, i, weight['ie']) for i in range(nI)]
    #         # weightMatrix = np.ones((nI, nE))
    #         # weightMatrix *= weight['ie']
    #         # for i in xrange(nI):
    #         #     weightMatrix[i,i] = 0
    #         # weightList = [(i, j, weightMatrix[i,j]) for i in xrange(nI) for j in xrange(nE)]
    #     else:
    #         weightMatrix = np.random.random((nI, nE))
    #         weightMatrix *= weight['ie']
    #         weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
    #     print 'save connection matrix', name
    #     np.save(dataPath + name, weightList)


if __name__ == "__main__":
    create_weights()

