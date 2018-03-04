# -*- coding:utf-8 -*-
"""
Created on 26.7.2017
Image from LGN reconstructed in V1 with four orientation
@author: Jiaxing Liu
"""
from parameters import *
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
# ------------------------------------------------------------------------------
# functions
# ------------------------------------------------------------------------------
def get_data(data_path):  # 加载方向条
    imgdata = glob(data_path + '*.jpg')
    imgs = []
    for i in range(len(imgdata)):
        temp_img = cv2.imread(imgdata[i],0)
        temp_img = imresize(temp_img, (input_size, input_size))
        imgs.append(temp_img)
    print('图像读取成功！总数：' + str(len(imgs)))
    return imgs

def get_matrix_from_file(fileName):  # 加载权值矩阵，初始化的权值矩阵以文件形式保存，需要的时候load进来
    offset = len(ending) + 4
    if fileName[-4 - offset] == 'X':  # 是否为输入，设置第一层个数 确定source和target层的神经元数量
        n_src = S1_cov_patch ** 2  # n_input
    else:
        if fileName[-3 - offset] == 'e':  # 兴奋性
            n_src = S1_num  # n_e
        else:
            n_src = S1_num  # n_i
    if fileName[-1 - offset] == 'e':
        n_tgt = S1_num  # n_e
    else:
        n_tgt = S1_num  # n_i
    readout = np.load(fileName)
    # print (readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))  # 全为0的权值矩阵
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]  # 将不为0的权值填充进去
    value_arr = (np.ones((S1_num, S1_cov_patch ** 2)) * value_arr[:, 1]).transpose()
    return value_arr

def get_init_matrix_from_file(fileName):  # 加载权值矩阵，初始化的权值矩阵以文件形式保存，需要的时候load进来
    offset = len(ending) + 4  # 读取文件名字，从后往前读 .npy
    if fileName[-8 - offset] == 'C':  # 是否为输入，设置第一层个数 确定source和target层的神经元数量
        n_src = S2_cov_patch ** 2
    else:
        pass
        # if fileName[-3 - offset] == 'e':  # 兴奋性
        #     n_src = n_e
        # else:
        #     n_src = n_i
    if fileName[-1 - offset] == 'e':
        n_tgt = S2_num
    else:
        n_tgt = S2_num
    readout = np.load(fileName)
    # print (readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))  # 全为0的权值矩阵
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]  # 将不为0的权值填充进去
    return value_arr

def normalize_weights():  # 权值归一化，用每一列的权值总和来归一化每一个权值
    # 类似于感受野内连接之间的侧抑制
    for connName in C1_S2_connections:  # XeAe XeBe XeCe XeDe
        # if connName[1] == 'e' and connName[3] == 'e':  # 兴奋到兴奋的连接，同个感受野内连接之间侧抑制
        connection = np.array(connections[connName].w / nS).reshape((S2_cov_patch_size, S2_num))
        # temp_conn = np.copy(connection)
        colSums = np.sum(connection, axis=0)  # 权值矩阵按列求和，即每个神经元感受野权值相加
        colFactors = weight['C1_S2_norm'] / colSums  # 期望总权值除以每一列的和
        for j in range(S2_num):  # n_e=25，每个兴奋性神经元与输入15*15神经元相连
            connection[:, j] *= colFactors[j]
            # np.array(connections[connName].w / nS).reshape((n_input, n_e))[:,j]*=colFactors[j]
        connections[connName].w = connection.reshape(S2_cov_patch_size * S2_num)[:] * nS

def get_2d_input_weights(layer):  # 将输入到兴奋层的连接转换为二维矩阵
    if layer == 'S1':
        connlist = input_S1_connections
        n_e_sqrt = S1_size  # int(np.sqrt(n_e))
        n_in_sqrt = S1_cov_patch  # int(np.sqrt(n_input))
        num_values_col = n_e_sqrt * n_in_sqrt
        num_values_row = num_values_col
        rearranged_weights = {}
        twoD_weights = np.zeros((2 * num_values_col, 2 * num_values_row))
        for name in connlist:
            rearranged_weights[name] = np.zeros((num_values_col, num_values_row))  # 105*105,将第二层每个兴奋性神经元的感受野二维排列出来
            connMatrix = np.array(connections[name].w / nS)
            weight_matrix = np.copy(connMatrix.reshape((S1_num, S1_cov_patch ** 2)).transpose())
            for i in range(n_e_sqrt):
                for j in range(n_e_sqrt):  # 每个神经元感受野权值赋值
                    rearranged_weights[name][i * n_in_sqrt: (i + 1) * n_in_sqrt, j * n_in_sqrt: (j + 1) * n_in_sqrt] = \
                        weight_matrix[:, i + j * n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
        for i in range(len(rearranged_weights) / 2):
            for j in range(len(rearranged_weights) / 2):
                # a= rearranged_weights[save_conns[i * 2 + j]]
                twoD_weights[i * num_values_col:(i + 1) * num_values_col, j * num_values_row:(j + 1) * num_values_row] = \
                    rearranged_weights[connlist[i * 2 + j]]  # 2*2排列四个网络的权值
        return twoD_weights  # 210*210
    elif layer == 'S2':
        connlist = C1_S2_connections[0:4]
        n_e_sqrt = S2_size  # int(np.sqrt(n_e))
        n_in_sqrt = S2_cov_patch  # int(np.sqrt(n_input))
        num_values_col = n_e_sqrt * n_in_sqrt
        num_values_row = num_values_col
        rearranged_weights = {}
        twoD_weights = np.zeros((2 * num_values_col, 2 * num_values_row))
        for name in connlist:
            rearranged_weights[name] = np.zeros((num_values_col, num_values_row))  # 105*105,将第二层每个兴奋性神经元的感受野二维排列出来
            connMatrix = np.array(connections[name].w / nS)
            weight_matrix = np.copy(connMatrix.reshape((S2_num, S2_cov_patch ** 2)).transpose())
            for i in range(n_e_sqrt):
                for j in range(n_e_sqrt):  # 每个神经元感受野权值赋值
                    rearranged_weights[name][i * n_in_sqrt: (i + 1) * n_in_sqrt, j * n_in_sqrt: (j + 1) * n_in_sqrt] = \
                        weight_matrix[:, i + j * n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
        for i in range(len(rearranged_weights) / 2):
            for j in range(len(rearranged_weights) / 2):
                # a= rearranged_weights[save_conns[i * 2 + j]]
                twoD_weights[i * num_values_col:(i + 1) * num_values_col, j * num_values_row:(j + 1) * num_values_row] = \
                    rearranged_weights[connlist[i * 2 + j]]
        return twoD_weights  # 210*210

def plot_2d_input_weights(layer):  # 将权重可视化出来
    weights = get_2d_input_weights(layer)
    # weights = np.ones((75, 75))
    fig = b2.figure(fig_num, figsize=(5, 5))
    im2 = b2.imshow(weights, interpolation="nearest", vmin=0, vmax=wmax_ee / nS, cmap=cmap.get_cmap('hot'))
    b2.colorbar(im2)  # 转换成伪彩图显示
    b2.title('weights of conv in ' + layer)
    fig.canvas.draw()
    return im2, fig

def update_2d_input_weights(im, fig, layer):  # 返回更新后的权值图案
    weights = get_2d_input_weights(layer)  # 二维权值矩阵 15*5 X 15*5
    im.set_array(weights)
    b2.pause(0.005)  # 以防界面卡死
    fig.canvas.draw()
    return im

def create_convolution_conn(imgsize, v1size, patch, padding):
    pre = []
    post = []
    if padding == 'valid':
        for row in np.arange(0 + np.floor(patch / 2), imgsize - np.floor(patch / 2)):  # row,col遍历后一层的行、列，去除边缘
            for col in np.arange(0 + np.floor(patch / 2), imgsize - np.floor(patch / 2)):
                for i in np.arange(row - np.floor(patch / 2), row + np.floor(patch / 2) + 1):  # i,j遍历前一层的行、列
                    for j in np.arange(col - np.floor(patch / 2), col + np.floor(patch / 2) + 1):
                        pre.append(int(i * imgsize + j))
                        post.append(int((row - np.floor(patch / 2)) * v1size + (col - np.floor(patch / 2))))
    return np.array(pre), np.array(post)  # 返回连接数组

def create_pooling_conn(s1size, c1size, patch, overlap):
    pre = []
    post = []
    stride = patch - overlap
    for row in np.arange(0, c1size):  # row,col遍历后一层的行、列，去除边缘
        for col in np.arange(0, c1size):
            for i in np.arange(row * stride, row * stride + patch):
                if (i >= s1size):
                    pass
                else:
                    for j in np.arange(col * stride, col * stride + patch):
                        if (j >= s1size):
                            pass
                        else:
                            pre.append(int(i * s1size + j))
                            post.append(int(row * c1size + col))
    return np.array(pre), np.array(post)  # 返回连接数组
    # post_i=0
    # post_j=0
    # i_stride = 0
    # j_stride = 0
    #             for i in np.arange(0+i_stride, 0 + 2*np.floor(patch / 2) + 1+i_stride):  # i,j遍历前一层的行、列
    #                 for j in np.arange(0+j_stride, 0 + 2*np.floor(patch / 2) + 1+j_stride):
    #                     pre.append(int(i * s1size + j))
    #                     post.append(int(row * c1size + col))
    #                     # if(int(row - np.floor(patch / 2))>0):
    #                     #     post_i = int(row - np.floor(patch / 2))-(patch-overlap)+1
    #                     # else:
    #                     #     post_i = int(row - np.floor(patch / 2))
    #                     # if (int(col - np.floor(patch / 2)) > 0):
    #                     #     post_j = int(col - np.floor(patch / 2)) - (patch - overlap)+1
    #                     # else:
    #                     #     post_j = int(col - np.floor(patch / 2))
    #             j_stride+=patch-overlap
    #         i_stride+=patch-overlap
    # return np.array(pre), np.array(post) # 返回连接数组

def create_pool_inh_conn(poolconn, c1size, patch):
    len = patch ** 2
    pre = []
    post = []
    for k in range(c1size ** 2):
        for i in poolconn[k * len:(k + 1) * len]:
            for j in poolconn[k * len:(k + 1) * len]:
                if (i == j):
                    pass
                else:
                    pre.append(i)
                    post.append(j)
    return np.array(pre), np.array(post)

def plot_2d_spike_rate(spike_count, layer):  # 将权重可视化出来
    if layer == 'S1':
        spk_mtx = np.reshape(spike_count, (S1_size, S1_size))
    if layer == 'C1':
        spk_mtx = np.reshape(spike_count, (C1_size, C1_size))
    if layer == 'S2':
        spk_mtx = np.reshape(spike_count, (S2_size, S2_size))
    fig = b2.figure(fig_num, figsize=(5, 5))
    im2 = b2.imshow(spk_mtx, interpolation="nearest", vmin=0, vmax=max(spike_count), cmap=cmap.get_cmap('jet'))
    b2.colorbar(im2)  # 转换成伪彩图显示
    b2.title('Spike count')
    fig.canvas.draw()
    return im2, fig

def update_2d_spike_rate(im, fig, spike_count,layer):  # 返回更新后的权值图案
    if layer == 'S1':
        spk_mtx = np.reshape(spike_count, (S1_size, S1_size))
    if layer == 'C1':
        spk_mtx = np.reshape(spike_count, (C1_size, C1_size))
    if layer == 'S2':
        spk_mtx = np.reshape(spike_count, (S2_size, S2_size))
    im.set_array(spk_mtx)
    b2.pause(0.005)  # 以防界面卡死
    fig.canvas.draw()
    return im

def save_2d_spike_rate(spike_count, layer):  # 将权重可视化出来
    if layer == 'S1':
        spk_mtx = np.reshape(spike_count, (S1_size, S1_size))
    if layer == 'C1':
        spk_mtx = np.reshape(spike_count, (C1_size, C1_size))
    if layer == 'S2':
        spk_mtx = np.reshape(spike_count, (S2_size, S2_size))
    if layer == 'C2':
        spk_mtx = np.reshape(spike_count, (C2_size, C2_size))
    # 显示画布
    fig = b2.figure(fig_num, figsize=(5, 5))
    # 显示图形，定义不同类型的colormap
    im2 = b2.imshow(spk_mtx, interpolation="nearest", vmin=0, vmax=max(spike_count), cmap=cmap.get_cmap('jet'))
    b2.colorbar(im2)  # 显示colorbar
    b2.title('Spike count')
    fig.savefig('cifar_edge/'+str(fig_num)+'.jpg')
    b2.close(fig)
# ------------------------------------------------------------------------------
# set parameters and equations
# ------------------------------------------------------------------------------
# 测试模式和训练模式下的参数
test_mode = 'True'
np.random.seed(0)
data_path = './'
if test_mode:  # 测试模式的参数
    weight_path = data_path + 'weights/'
    num_examples = 10000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False  # 不需要学习
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'
    num_examples = 500 * 1  # 训练样本集60000重复训练3次
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 60000:
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True  # 输入到兴奋层通过STDP学习

ending = ''  # 路径结尾的名称
single_example_time = 0.35 * second  # 每个样本训练时间为350ms，然后切换下一张图像
resting_time = 0.15 * second  # 间隔150ms，使与时间有关的参数回归初始值，例如电导ge、内稳态变量theta等

# runtime = num_examples * (single_example_time + resting_time)
# if num_examples <= 10000:
#     update_interval = num_examples
#     weight_update_interval = 10
# else:  # 训练
#     update_interval = 10000
#     weight_update_interval = 100
# if num_examples <= 60000:
#     save_connections_interval = 10000
# else:
#     save_connections_interval = 10000
#     update_interval = 10000
# 设置参数
# 神经元群名称
input_population_names = ['X']  # 输入层名字，代表LGN
S1_population_names = ['A', 'B', 'C', 'D']  # 兴奋层 应该有四个 A，B，C，D
S2_population_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# 层连接名称
input_S1_conn_names = ['XA', 'XB', 'XC', 'XD']  # 输入到兴奋的连接
C1_S2_conn_names = ['AA', 'BA', 'CA', 'DA', 'AB', 'BB', 'CB', 'DB', 'AC', 'BC', 'CC', 'DC', 'AD', 'BD', 'CD', 'DD',
                    'AE', 'BE', 'CE', 'DE', 'AF', 'BF', 'CF', 'DF','AG', 'BG', 'CG', 'DG','AH', 'BH', 'CH', 'DH']
# 层连接类型
input_S1_conn_types = ['ee']  # 输入到兴奋的连接为ee
S1_recurrent_conn_types = ['ei', 'ie']  # S1兴奋到抑制的连接
C1_pooling_conn_types = ['ee', 'ei', 'ie']
C1_S2_conn_types = ['ee']
S2_recurrent_conn_types = ['ei', 'ie']  # S2与其抑制层的连接
S2_C2_conn_types = ['ee']

# 卷积层连接
input_S1_connections = ['XeAe', 'XeBe', 'XeCe', 'XeDe']  # 输入到S1的连接
C1_S2_connections = ['C1AeS2Ae', 'C1BeS2Ae', 'C1CeS2Ae', 'C1DeS2Ae', 'C1AeS2Be', 'C1BeS2Be', 'C1CeS2Be', 'C1DeS2Be',
                     'C1AeS2Ce', 'C1BeS2Ce', 'C1CeS2Ce', 'C1DeS2Ce', 'C1AeS2De', 'C1BeS2De', 'C1CeS2De', 'C1DeS2De',
                     'C1AeS2Ee', 'C1BeS2Ee', 'C1CeS2Ee', 'C1DeS2Ee', 'C1AeS2Fe', 'C1BeS2Fe', 'C1CeS2Fe', 'C1DeS2Fe',
                     'C1AeS2Ge', 'C1BeS2Ge', 'C1CeS2Ge', 'C1DeS2Ge', 'C1AeS2He', 'C1BeS2He', 'C1CeS2He', 'C1DeS2He']

# 设置发放阈值，自适应阈值，稳态一部分
# if test_mode:
#     scr_e = 'v = v_reset_e'
# else:
#     tc_theta = 1e5 * ms  # 阈值参数theta衰减的很慢
#     theta_plus_e = 0.0005 * mV  # 每发放一个脉冲，阈值增加plus_e 阈值增加的快慢
#     scr_e = 'v = v_reset_e; theta += theta_plus_e'  # 每发放一次，theta就改变一次
# offset = 20.0 * mV
# 兴奋性神经元膜电位时间常数为50ms 兴奋性电导时间常数为1ms
tc_theta = 1e6 * ms  # 阈值参数theta衰减的很慢
theta_plus_e = 0.01 * mV  # 每发放一个脉冲，阈值增加plus_e 阈值增加的快慢
res_e = 'v = v_reset_e; theta += theta_plus_e'  # 每发放一次，theta就改变一次
offset = 20.0 * mV
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / gleak) / taum_e  : volt (unless refractory)
        I_synE = ge *         -v                                : amp
        I_synI = gi * (-100.*mV-v)                              : amp
        dge/dt = -ge/(1.0*ms)                                   : siemens
        dgi/dt = -gi/(2.0*ms)                                   : siemens
        dtheta/dt = -theta / (tc_theta)                         : volt
      '''
# if test_mode:
#     neuron_eqs_e += '\n  theta      :volt'
# else:
#     neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
# 抑制性神经元的时间常数小，发放快
neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / gleak) / taum_i  : volt (unless refractory)
        I_synE = ge *         -v                                : amp
        I_synI = gi * (-100.*mV-v)                              : amp
        dge/dt = -ge/(1.0*ms)                                   : siemens
        dgi/dt = -gi/(2.0*ms)                                   : siemens
      '''
# 输入神经元 将像素转为脉冲序列
poisson_neurons = '''
rates : Hz
dv/dt = 1 : second'''
# max-pooling神经元
neuron_max_pool = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / gleak) / taum_e  : volt (unless refractory)
        I_synE = ge *         -v                                : amp
        I_synI = gi * (-100.*mV-v)                               : amp
        dge/dt = -ge/(1.0*ms)                                   : siemens
        dgi/dt = -gi/(2.75*ms)                                   : siemens
      '''
# determine STDP rule to use，确定使用哪种STDP规则
stdp_input = ''
use_weight_dependence = False
stdp_input += 'no_weight_dependence_'

post_pre = True
stdp_input += 'postpre'
# if raw_input('Use weight dependence (default no)?: ') in ['no', '']:
#     use_weight_dependence = False
#     stdp_input += 'no_weight_dependence_'
# else:
#     use_weight_dependence = True
#     stdp_input += 'weight_dependence_'
#
# if raw_input('Enter (yes / no) for post-pre (default yes): ') in ['yes', '']:
#     post_pre = True
#     stdp_input += 'postpre'
# else:
#     post_pre = False
#     stdp_input += 'no_postpre'

# STDP synaptic traces，STDP随时间变换的公式
# 可以把pre、post看作是突触的变量，随时间变换
eqs_stdp_ee = '''
            w:siemens
            dpre/dt = -pre / tc_pre_ee : siemens (event-driven)
            dpost/dt = -post / tc_post_ee : siemens (event-driven)
            '''
# setting STDP update rule 四种改进的STDP更新规则
if use_weight_dependence:
    if post_pre:
        eqs_stdp_pre_ee = '''
        ge+=w
        pre = 1.*nS
        w = clip(w - nu_ee_pre * post * w ** exp_ee_pre, 0, wmax_ee)
        '''
        eqs_stdp_post_ee = '''
        w = clip(w + nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post, 0, wmax_ee)
        post = 1.*nS
        '''

    else:
        eqs_stdp_pre_ee = '''
        ge+=w
        pre = 1.*nS
        '''
        eqs_stdp_post_ee = '''
        w = clip(w + nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post, 0, wmax_ee)
        post = 1.*nS
        '''

else:
    if post_pre:
        eqs_stdp_pre_ee = '''
        ge+=w
        pre = 1.*nS
        w = clip(w - nu_ee_pre * post, 0, wmax_ee)
        '''
        eqs_stdp_post_ee = '''
        w = clip(w + nu_ee_post * pre, 0, wmax_ee)
        post = 1.*nS
        '''

    else:
        eqs_stdp_pre_ee = '''
        ge+=w
        pre = 1.*nS
        '''
        eqs_stdp_post_ee = '''
        w = clip(w + nu_ee_post * pre, 0, wmax_ee)
        post = 1.*nS
        '''
# b2.ion()
fig_num = 1

# result_monitor = np.zeros((update_interval, n_e))  # 10000*25，每个神经元对样本的反应
# 兴奋层有四组神经元，对应四个方向功能柱，15*15*4=900
S1_groups['e'] = b2.NeuronGroup(S1_num * len(S1_population_names), model=neuron_eqs_e,
                                threshold='v>(- 52*mV)', refractory=refrac_e,
                                reset='v = v_reset_e', method='euler')  # 阈值决定了特征图中特征的明显程度，阈值越大，特征越突出明显
# S1_groups['e'] = b2.NeuronGroup(S1_num* len(S1_population_names), model=neuron_eqs_e, threshold='v>(theta - offset - 52*mV)',refractory=refrac_e,
#                                    reset=scr_e, method='euler')
S1_groups['i'] = b2.NeuronGroup(S1_num * len(S1_population_names), model=neuron_eqs_i, threshold='v>v_thresh_i',
                                refractory=refrac_i,
                                reset='v=v_reset_i', method='euler')
C1_groups['e'] = b2.NeuronGroup(C1_num * len(S1_population_names), model=neuron_eqs_e, threshold='v>v_thresh_e',
                                refractory=refrac_e,
                                reset='v=v_reset_e', method='euler')
pool1_groups['e'] = b2.NeuronGroup(S1_num * len(S1_population_names), model=neuron_eqs_e, threshold='v>v_thresh_e',
                                   refractory=refrac_e,
                                   reset='v=v_reset_e', method='euler')
pool1_groups['i'] = b2.NeuronGroup(S1_num * len(S1_population_names), model=neuron_eqs_i, threshold='v>v_thresh_i',
                                   refractory=refrac_i,
                                   reset='v=v_reset_i', method='euler')
S2_groups['e'] = b2.NeuronGroup(S2_num * len(S2_population_names), model=neuron_eqs_e, threshold='v>(theta - offset - 52*mV)',
                                refractory=refrac_e,
                                reset=res_e, method='euler')
S2_groups['i'] = b2.NeuronGroup(S2_num * len(S2_population_names), model=neuron_eqs_i, threshold='v>v_thresh_i',
                                refractory=refrac_i,
                                reset='v=v_reset_i', method='euler')
C2_groups['e'] = b2.NeuronGroup(C2_num * len(S2_population_names), model=neuron_eqs_e, threshold='v>v_thresh_e',
                                refractory=refrac_e,
                                reset='v=v_reset_e', method='euler')
# ------------------------------------------------------------------------------
# create S1 and C1 population and connections
# ------------------------------------------------------------------------------
# 建立S1与C1的突触连接

pre_s1, post_c1 = create_pooling_conn(S1_size, C1_size, C1_pool_patch, 0)  # pooling的patch大小尽量是奇数，并且重叠越少越好
pre_pool, post_pool = create_pool_inh_conn(pre_s1, C1_size, C1_pool_patch)
for idx, name in enumerate(S1_population_names):  # 创建若干神经元子群 'A' 'B' 'C' 'D'
    print('--------创建神经元群S1、C1' + name + '--------')
    S1_groups[name + 'e'] = S1_groups['e'][idx * S1_num:(idx + 1) * S1_num]  # neuron_groups['Ae']
    S1_groups[name + 'i'] = S1_groups['i'][idx * S1_num:(idx + 1) * S1_num]  # neuron_groups['Ai']
    C1_groups[name + 'e'] = C1_groups['e'][idx * C1_num:(idx + 1) * C1_num]
    pool1_groups[name + 'e'] = pool1_groups['e'][idx * S1_num:(idx + 1) * S1_num]
    pool1_groups[name + 'i'] = pool1_groups['i'][idx * S1_num:(idx + 1) * S1_num]

    S1_groups[name + 'e'].v = v_rest_e - 10. * mV
    S1_groups[name + 'i'].v = v_rest_i - 10. * mV

    S1_groups[name + 'e'].ge = 0 * nS
    S1_groups[name + 'e'].gi = 0 * nS
    S1_groups[name + 'i'].ge = 0 * nS
    S1_groups[name + 'i'].gi = 0 * nS

    C1_groups[name + 'e'].v = v_rest_e - 10. * mV
    C1_groups[name + 'e'].ge = 0 * nS
    C1_groups[name + 'e'].gi = 0 * nS

    pool1_groups[name + 'e'].v = v_rest_e - 10. * mV
    pool1_groups[name + 'e'].ge = 0 * nS
    pool1_groups[name + 'e'].gi = 0 * nS

    pool1_groups[name + 'i'].v = v_rest_i - 10. * mV
    pool1_groups[name + 'i'].ge = 0 * nS
    pool1_groups[name + 'i'].gi = 0 * nS
    # if test_mode or weight_path[-8:] == 'weights/':  # 加载阈值参数值
    # neuron_groups[name + 'e'].theta = np.load(weight_path + 'theta_' + name + '2017.7.26' + ending + '.npy')
    # else:
    #S1_groups[name + 'e'].theta = np.ones((S1_num)) * 20.0 * mV  # 训练时，theta的初始值为20mV

    print ('S1层兴奋与抑制连接')  # 建立兴奋与抑制的连接
    conn_type = S1_recurrent_conn_types[0]  # 'ei'
    connName = 'S1' + name + conn_type[0] + 'S1' + name + conn_type[1]  # AeAi AiAe BeBi BiBe
    # weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')  # 初始化权值
    if (conn_type == 'ei'):
        connections[connName] = b2.Synapses(S1_groups[connName[2:4]], S1_groups[connName[6:8]],
                                            model='w:siemens', on_pre='ge+=w', method='euler')  # conn_type='ei'/'ie'
        connections[connName].connect(j='i')  # 建立一对一连接
        # connections[connName].w = weightMatrix.reshape((v1_num * v1_num)) * nS
        connections[connName].w = np.ones(S1_num) * 10. * nS

    print('S1层与pool_e层连接')  # 建立S1与pool的连接，pool神经元为中间复制层，完成非线性转换
    pool_conn_type = C1_pooling_conn_types[0]  # 'ee'
    pool_connName = 'S1' + name + pool_conn_type[0] + 'p1' + name + pool_conn_type[1]
    connections[pool_connName] = b2.Synapses(S1_groups[pool_connName[2:4]], pool1_groups[pool_connName[6:8]],
                                             model='w:siemens', on_pre='ge+=w', method='euler')
    connections[pool_connName].connect(j='i')
    connections[pool_connName].w = weight['S1_pool1']
    print('中间pool_e层与C1层连接')  # 建立pool与C1的连接，max-pool接收最强输入
    pool_conn_type = C1_pooling_conn_types[0]  # 'ee'
    pool_connName = 'p1' + name + pool_conn_type[0] + 'C1' + name + pool_conn_type[1]
    connections[pool_connName] = b2.Synapses(pool1_groups[pool_connName[2:4]], C1_groups[pool_connName[6:8]],
                                             model='w:siemens', on_pre='ge+=w', method='euler')
    connections[pool_connName].connect(i=pre_s1, j=post_c1)
    connections[pool_connName].w = weight['pool1_C1']

    print('中间pool_e层与pool_i层连接')  # 建立pool与C1的连接，max-pool接收最强输入
    pool_conn_type = C1_pooling_conn_types[1]  # 'ei'
    pool_connName = 'p1' + name + pool_conn_type[0] + 'p1' + name + pool_conn_type[1]
    connections[pool_connName] = b2.Synapses(pool1_groups[pool_connName[2:4]], pool1_groups[pool_connName[6:8]],
                                             model='w:siemens', on_pre='ge+=w', method='euler')
    connections[pool_connName].connect(j='i')
    connections[pool_connName].w = weight['pool1E_I']

    print('pool_i与pool_e的连接，实现max-pooling')  # 建立pool自己的抑制连接，实现max-pool
    pool_conn_type = C1_pooling_conn_types[2]  # 'ie'
    pool_connName = 'p1' + name + pool_conn_type[0] + 'p1' + name + pool_conn_type[1]
    connections[pool_connName] = b2.Synapses(pool1_groups[pool_connName[2:4]], pool1_groups[pool_connName[6:8]],
                                             model='w:siemens', on_pre='gi+=w', method='euler')
    connections[pool_connName].connect(i=pre_pool, j=post_pool)
    connections[pool_connName].w = weight['pool1I_E']

    spike_monitors['p1' + name + 'e'] = b2.SpikeMonitor(pool1_groups[name + 'e'])
    spike_monitors['C1' + name + 'e'] = b2.SpikeMonitor(C1_groups[name + 'e'])

# 建立抑制连接，每个抑制性神经元与同一位置的其他三种神经元连接
print('S1层抑制与兴奋的连接')
for name in S1_population_names:  # A B C D
    # for name2 in S1_population_names:
    #     conn_type = S1_recurrent_conn_types[1]  # 'ie'
    #     if name2 == name:
    #         pass
    #     else:
    #         connName = 'S1' + name + conn_type[0] + 'S1' + name2 + conn_type[1]
    #         # weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')  # 加载初始权值
    #         if (conn_type == 'ie'):
    #             connections[connName] = b2.Synapses(S1_groups[connName[2:4]], S1_groups[connName[6:8]],
    #                                                 model='w:siemens', on_pre='gi+=w', method='euler')  # conn_type='ie'
    #             connections[connName].connect(j='i')  # 建立全连接
    #             # connections[connName].w = weightMatrix.reshape((v1_num * v1_num)) * nS
    #             connections[connName].w = np.ones(S1_num) * 17. * nS

    print ('create spike monitors for', name)
    if record_spikes:
        spike_monitors['S1' + name + 'e'] = b2.SpikeMonitor(S1_groups[name + 'e'])
        spike_monitors['S1' + name + 'i'] = b2.SpikeMonitor(S1_groups[name + 'i'])
# ------------------------------------------------------------------------------
# create S2 and C2 population and connections
# ------------------------------------------------------------------------------
pre_s2,post_c2 = create_pooling_conn(S2_size,C2_size,C2_pool_patch,0) # pooling的patch大小尽量是奇数，并且重叠越少越好
# pre_pool,post_pool = create_pool_inh_conn(pre_s1,C1_size,C1_pool_patch)
for idx, name in enumerate(S2_population_names):  # 创建若干神经元子群 'A' 'B' 'C' 'D' 'E' 'F'
    print('--------创建神经元群S2' + name + '--------')
    S2_groups[name + 'e'] = S2_groups['e'][idx * S2_num:(idx + 1) * S2_num]  # neuron_groups['Ae']
    S2_groups[name + 'i'] = S2_groups['i'][idx * S2_num:(idx + 1) * S2_num]  # neuron_groups['Ai']
    C2_groups[name + 'e'] = C2_groups['e'][idx * C2_num:(idx + 1) * C2_num]  # ABCDEFGH
    # pool2_groups[name + 'e'] = pool2_groups['e'][idx*S2_num:(idx+1)*S2_num]

    # 设置神经元群变量属性
    S2_groups[name + 'e'].v = v_rest_e - 10. * mV
    S2_groups[name + 'i'].v = v_rest_i - 10. * mV

    S2_groups[name + 'e'].ge = 0 * nS
    S2_groups[name + 'e'].gi = 0 * nS
    S2_groups[name + 'i'].ge = 0 * nS
    S2_groups[name + 'i'].gi = 0 * nS

    C2_groups[name + 'e'].v = v_rest_e - 10. * mV
    C2_groups[name + 'e'].ge = 0 * nS
    C2_groups[name + 'e'].gi = 0 * nS
    #
    # pool2_groups[name + 'e'].v = v_rest_e - 10. * mV
    # pool2_groups[name + 'e'].ge = 0 * nS
    # pool2_groups[name + 'e'].gi = 0 * nS
    # if test_mode or weight_path[-8:] == 'weights/':  # 加载阈值参数值
    # neuron_groups[name + 'e'].theta = np.load(weight_path + 'theta_' + name + '2017.7.26' + ending + '.npy')
    # else:
    S2_groups[name + 'e'].theta = np.ones((S2_num)) * 20.0 * mV  # 训练时，theta的初始值为20mV，发放脉冲，theta改变从而阈值改变

    # S2层与S1层一样，引入竞争机制，同一位置的不同特征神经元相互抑制，学到不同的组合特征
    print ('S2层兴奋与抑制连接')  # 建立兴奋与抑制的连接
    conn_type = S2_recurrent_conn_types[0]  # 'ei'
    connName = 'S2' + name + conn_type[0] + 'S2' + name + conn_type[1]  # AeAi BeBi
    # weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')  # 初始化权值
    if (conn_type == 'ei'):
        connections[connName] = b2.Synapses(S2_groups[connName[2:4]], S2_groups[connName[6:8]],
                                            model='w:siemens', on_pre='ge+=w', method='euler')  # conn_type='ei'/'ie'
        connections[connName].connect(j='i')  # 建立一对一连接
        # connections[connName].w = weightMatrix.reshape((v1_num * v1_num)) * nS
        connections[connName].w = np.ones(S2_num) * weight['S2E_I']
    print('S2层与C2层连接')  # 建立pool与C1的连接，max-pool接收最强输入
    pool_conn_type = S2_C2_conn_types[0]  # 'ee'
    pool_connName = 'S2' + name + pool_conn_type[0] + 'C2' + name + pool_conn_type[1]
    connections[pool_connName] = b2.Synapses(S2_groups[pool_connName[2:4]], C2_groups[pool_connName[6:8]],
                                             model='w:siemens', on_pre='ge+=w', method='euler')
    connections[pool_connName].connect(i=pre_s2, j=post_c2)
    connections[pool_connName].w = weight['S2_C2']
    spike_monitors['C2' + name + 'e'] = b2.SpikeMonitor(C2_groups[name + 'e'])
        # print('create S1-pooling connections') # 建立S2与pool2的连接，pool2神经元为中间复制层，完成非线性转换
        # pool_conn_type = C1_pooling_conn_types[0] # 'ee'
        # pool_connName = 'S'+name+pool_conn_type[0]+'p'+name+pool_conn_type[1]
        # connections[pool_connName] = b2.Synapses(S1_groups[pool_connName[1:3]], pool_groups[pool_connName[4:6]],
        #                                          model='w:siemens', on_pre='ge+=w',method='euler')
        # connections[pool_connName].connect(j='i')
        # connections[pool_connName].w = weight['S1_pool']
        #
        # print('create pooling-C1 connections')  # 建立pool2与C2的连接，max-pool接收最强输入
        # pool_conn_type = C1_pooling_conn_types[0]  # 'ee'
        # pool_connName = 'p' + name + pool_conn_type[0] + 'C' + name + pool_conn_type[1]
        # connections[pool_connName] = b2.Synapses(pool_groups[pool_connName[1:3]], C1_groups[pool_connName[4:6]],
        #                                          model='w:siemens', on_pre='ge+=w', method='euler')
        # connections[pool_connName].connect(i=pre_s1,j=post_c1)
        # connections[pool_connName].w = weight['pool_C1']
        #
        # spike_monitors['p' + name + 'e'] = b2.SpikeMonitor(pool_groups[name + 'e'])
        # spike_monitors['C' + name + 'e'] = b2.SpikeMonitor(C1_groups[name + 'e'])
        #
        # print('create pooling-inh connections')  # 建立pool2自己的抑制连接，实现max-pool
        # pool_conn_type = C1_pooling_conn_types[0]  # 'ee'
        # pool_connName = 'p' + name + pool_conn_type[0] + 'p' + name + pool_conn_type[1]
        # connections[pool_connName] = b2.Synapses(pool_groups[pool_connName[1:3]], pool_groups[pool_connName[4:6]],
        #                                          model='w:siemens', on_pre='gi+=w', method='euler')
        # connections[pool_connName].connect(i=pre_pool,j=post_pool)
        # connections[pool_connName].w = weight['pool_Inh']
        #
        # spike_monitors['p' + name + 'e'] = b2.SpikeMonitor(pool_groups[name + 'e'])
        # spike_monitors['C' + name + 'e'] = b2.SpikeMonitor(C1_groups[name + 'e'])

# 建立抑制连接，每个抑制性神经元与同一位置的其他三种神经元连接
print ('S2层抑制与兴奋连接')  # 建立兴奋与抑制的连接
for name1 in S2_population_names:  # A B C D E F
    for name2 in S2_population_names:
        conn_type = S2_recurrent_conn_types[1]  # 'ie'
        if name2 == name1:
            # pass
            connName = 'S2' + name1 + conn_type[0] + 'S2' + name2 + conn_type[1]
            if (conn_type == 'ie'):
                connections[connName] = b2.Synapses(S2_groups[connName[2:4]], S2_groups[connName[6:8]],
                                                    model='w:siemens', on_pre='gi+=w', method='euler')  # conn_type='ie'
                connections[connName].connect(condition='i!=j')  # 建立全连接
                connections[connName].w = weight['S2I_E']
        else:
            connName = 'S2' + name1 + conn_type[0] + 'S2' + name2 + conn_type[1]
            # weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')  # 加载初始权值
            if (conn_type == 'ie'):
                connections[connName] = b2.Synapses(S2_groups[connName[2:4]], S2_groups[connName[6:8]],
                                                    model='w:siemens', on_pre='gi+=w', method='euler')  # conn_type='ie'
                connections[connName].connect(j='i')  # 建立全连接
                connections[connName].w = weight['S2IE_single']

for name in S2_population_names:  # A B C D E F
    print ('create monitors for', name)  # monitors for A
    # rate_monitors[name + 'e'] = b2.PopulationRateMonitor(S2_groups[name + 'e'])  # 350ms+150ms
    # rate_monitors[name + 'i'] = b2.PopulationRateMonitor(S2_groups[name + 'i'])

    if record_spikes:
        spike_monitors['S2' + name + 'e'] = b2.SpikeMonitor(S2_groups[name + 'e'])
        spike_monitors['S2' + name + 'i'] = b2.SpikeMonitor(S2_groups[name + 'i'])

temp=S2_groups['Ae']
stateS2Ae = b2.StateMonitor(temp,('v','I_synI'),record=[0])
temp=S2_groups['Ai']
stateS2Ai = b2.StateMonitor(temp,('v','I_synE'),record=[0])
# ------------------------------------------------------------------------------
# create connections between C1 and S2
# ------------------------------------------------------------------------------
# pop_values = ['A', 'B', 'C','D']  # 创建输入到兴奋层的连接
# spk_len = {} # 记录S2每个网络发放神经元的个数
print('建立C1层与S2层连接')
pre_C1, post_S2 = create_convolution_conn(C1_size, S2_size, S2_cov_patch, S2_cov_padding)
for name in C1_S2_conn_names:  # 'AA','BA','CA','DA','AB','BB','CB','DB','AC','BC','CC','DC','AD','BD','CD','DD'
    # print ('建立C1' + name[0] + '与S2' + name[1] + '的连接')
    for connType in C1_S2_conn_types:  # 'ee'
        connName = 'C1' + name[0] + connType[0] + 'S2' + name[1] + connType[1]  # connName='C1AeS2Ae' 'C1BeS2Ae'
        # spk_len[connName] = 0
        weightMatrix = get_init_matrix_from_file(data_path + 'random/' + connName + ending + '.npy')  # 加载初始权值矩阵
        # weightMatrix = get_matrix_from_file(weight_path +'2017.8.2_55/'+ connName + ending + '.npy')  # 加载初始权值矩阵
        connections[connName] = b2.Synapses(C1_groups[connName[2:4]], S2_groups[connName[6:8]], model=eqs_stdp_ee,
                                            on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee, method='euler')
        # connections[connName] = b2.Synapses(C1_groups[connName[2:4]], S2_groups[connName[6:8]], model='w:siemens',
        #                                     on_pre='ge+=w', method='euler')
        connections[connName].connect(i=pre_C1, j=post_S2)  # 全连接
        connections[connName].delay = 0 * ms  # delay[connType]  # 突触延迟是一个范围，0-10之间
        connections[connName].w = weightMatrix.transpose().reshape(
            (S2_cov_patch ** 2 * S2_num)) * nS  # weightMatrix.reshape((25*26*26))*siemens
# print(connections['XeAe'].w) # 初始的第一层权值
# print(connections['XeBe'].w) # 初始的第一层权值
# print(connections['XeCe'].w) # 初始的第一层权值
# print(connections['XeDe'].w) # 初始的第一层权值
# ------------------------------------------------------------------------------
# create input population and connections from input populations
# ------------------------------------------------------------------------------
# 创建输入到兴奋层的连接
pop_values = ['A', 'B', 'C', 'D']
for i, name in enumerate(input_population_names):  # 'X'
    input_groups[name + 'e'] = NeuronGroup(input_size * input_size, model=poisson_neurons, threshold='v>1/rates',
                                           reset='v=0*second', method='euler')
    # b2.PoissonGroup(img_size*img_size, 0*Hz)  # input_groups['Xe'] n_input=15*15
    rate_monitors[name + 'e'] = b2.PopulationRateMonitor(input_groups[name + 'e'])
pre_input, post_S1 = create_convolution_conn(input_size, S1_size, S1_cov_patch, S1_cov_padding)
for name in input_S1_conn_names:  # 'XA' 'XB' 'XC' 'XD'
    print ('----建立' + name[0] + '与S1' + name[1] + '的连接----')
    for connType in input_S1_conn_types:  # 'ee'
        connName = name[0] + connType[0] + name[1] +\
                   connType[1]  # connName='XeAe' 'XeBe' 'XeCe' 'XeDe'
        weightMatrix = get_matrix_from_file(weight_path + '2017.7.26_77/' + connName + ending + '.npy')  # 加载初始权值矩阵
        # connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]], model=eqs_stdp_ee,
        #                                     on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee, method='euler')
        connections[connName] = b2.Synapses(input_groups[connName[0:2]], S1_groups[connName[2:4]],
                                            model='w:siemens', on_pre='ge+=w', method='euler') # 没有STDP学习
        connections[connName].connect(i=pre_input, j=post_S1)  # 全连接
        connections[connName].delay = 0 * ms  # delay[connType]  # 突触延迟是一个范围，0-10之间
        connections[connName].w = weightMatrix.transpose().reshape(
            (S1_cov_patch ** 2 * S1_size ** 2)) * nS  # weightMatrix.reshape((49*225))*siemens

# print(connections['XeAe'].w) # 初始的第一层权值
# print(connections['XeBe'].w) # 初始的第一层权值
# print(connections['XeCe'].w) # 初始的第一层权值
# print(connections['XeDe'].w) # 初始的第一层权值

# ------------------------------------------------------------------------------
# run the simulation and set inputs
# ------------------------------------------------------------------------------
previous_spike_count = np.zeros(S1_num)  # 25 记录每个神经元脉冲数量

# if not test_mode:  # 可视化训练的权值
# input_S1_weights, input_S1_fig = plot_2d_input_weights('S1')  # 神经元群感受野可视化
# fig_num += 1
# C1_S2_weights, C1_S2_fig = plot_2d_input_weights('S2')
# fig_num += 1
for i, name in enumerate(input_population_names):  # 'X'
    input_groups[name + 'e'].rates = 0 * Hz
# 将所有的容器add进来
net = Network(collect())
net.add(S1_groups)
net.add(C1_groups)
net.add(input_groups)
net.add(pool1_groups)
net.add(S2_groups)
net.add(C2_groups)
net.add(connections)
net.add(spike_monitors)
net.run(0 * ms)

# 记录神经元群发放脉冲数量
sc1 = spike_monitors['S1Ae'].count
sc2 = spike_monitors['S1Be'].count
sc3 = spike_monitors['S1Ce'].count
sc4 = spike_monitors['S1De'].count

scp1 = spike_monitors['p1Ae'].count
scp2 = spike_monitors['p1Be'].count
scp3 = spike_monitors['p1Ce'].count
scp4 = spike_monitors['p1De'].count

scc1 = spike_monitors['C1Ae'].count
scc2 = spike_monitors['C1Be'].count
scc3 = spike_monitors['C1Ce'].count
scc4 = spike_monitors['C1De'].count

s21 = spike_monitors['S2Ae'].count
s22 = spike_monitors['S2Be'].count
s23 = spike_monitors['S2Ce'].count
s24 = spike_monitors['S2De'].count

# img1,fig1 = plot_2d_spike_rate(np.array(scc1), 'C1')
# fig_num += 1
# img2,fig2 = plot_2d_spike_rate(np.array(s21), 'S2')
# fig_num += 1
# img3,fig3 = plot_2d_spike_rate(np.array(s22), 'S2')
# fig_num += 1
# img4,fig4 = plot_2d_spike_rate(np.array(s23), 'S2')
# fig_num += 1
# img5,fig5 = plot_2d_spike_rate(np.array(s24), 'S2')
# fig_num += 1

# 获取图像
img_cifar = get_data('./cifar_edge/') # 读取训练图像
# img_face = get_data('images/train_edge/face_easy/')
# img_motor = get_data('images/train_edge/motor/')
# img_motor = get_data('./') # 读入图像
training_start = time.time()
pre_count = {}
cur_count = {}
for name in S2_population_names:
    pre_count['S2'+name+'e'] = np.zeros(S2_num)
    pre_count['C2' + name + 'e'] = np.zeros(C2_num)

pre_count1 = np.zeros(S1_num)
pre_count2 = np.zeros(S1_num)
pre_count3 = np.zeros(S1_num)
pre_count4 = np.zeros(S1_num)
pre_count21 = np.zeros(C1_num)
j = 0 # 训练图像索引
p = 0
q = 0
idx = {}
#while j < (len(img_face) + len(img_motor)):  # 大循环，每个训练样本跑一次
while j<(1):
    normalize_weights()  # 权值归一化，稳态机制
    # i = j%80
    # if i<=20:
    #     rate = img[0].reshape((n_input)) / 8. * input_intensity
    # if(i>20 and i<=40):
    #     rate = img[1].reshape((n_input)) / 8. * input_intensity
    # if (i>40 and i<=60):
    #     rate = img[2].reshape((n_input)) / 8. * input_intensity
    # if (i>60):
    #     rate = img[3].reshape((n_input)) / 8. * input_intensity
    # i = j % 40
    # if i <= 10:
    #     rate = img[0].reshape((n_input)) / 8. * input_intensity
    # if (i > 10 and i <= 20):
    #     rate = img[1].reshape((n_input)) / 8. * input_intensity
    # if (i > 20 and i <= 30):
    #     rate = img[2].reshape((n_input)) / 8. * input_intensity
    # if (i > 30):
    #     rate = img[3].reshape((n_input)) / 8. * input_intensity
    # update_2d_input_weights(C1_S2_weights, C1_S2_fig, 'S2')
    # i = j % 4
    # if i < 2:
    #     rate = img_face[p].reshape((input_num)) / 8. * input_intensity
    #     p+=1
    # if i >= 2:
    #     rate = img_motor[q].reshape((input_num)) / 8. * input_intensity
    #     q+=1
    rate = img_cifar[j].reshape((input_num)) / 8. * input_intensity
    input_groups['Xe'].rates = rate * Hz # 第一层poisson神经元赋频率
    # print 'run number:', j + 1, 'of', len(img_motor) + len(img_motor) # 打印目前训练到哪张图片
    net.run(single_example_time, report='text')  # 每个样本运行350ms
    # print('图片跑完')
    current_spike_count = np.asarray(spike_monitors['S1Ae'].count[:] + spike_monitors['S1Be'].count[:] +
                                     spike_monitors['S1Ce'].count[:] + spike_monitors['S1De'].count[:]) \
                          - previous_spike_count  # 记录Ae每个神经元发放的脉冲数
    # if j % weight_update_interval == 0 and not test_mode:  # 每100个样本更新一次权重
    #     update_2d_input_weights(input_weight_monitor, fig_weights)
    #     print('weights < 0.2 :',len(np.where(connections['XeAe'].w / nS < 0.2)[0]))
    #     print(neuron_groups['Ae'].theta - offset - 52 * mV)
    #     print('spike count', current_spike_count)
    # if j % save_connections_interval == 0 and j > 0 and not test_mode:  # 间隔保存权值和theta参数
    #     save_connections(str(j))
    #     save_theta(str(j))

    previous_spike_count = np.copy(spike_monitors['S1Ae'].count[:] + spike_monitors['S1Be'].count[:] +
                                   spike_monitors['S1Ce'].count[:] + spike_monitors['S1De'].count[:])
    # print('rest')
    if np.sum(current_spike_count) < 20:  # 如果总脉冲数小于20
        input_intensity += 1  # 输入像素转换的脉冲频率大一倍
        for i, name in enumerate(input_population_names):  # 'X'
            input_groups[name + 'e'].rates = 0 * Hz
        net.run(resting_time,report='text')  # 暂停输入，使参数静息
    else:
        for i, name in enumerate(input_population_names):  # 'X'
            input_groups[name + 'e'].rates = 0 * Hz  # 暂停输入
        net.run(resting_time) # 休息150ms，使参数回归初始值
        input_intensity = start_input_intensity
        # update_2d_spike_rate(img1, fig1,np.array(scc1),'C1')
        # update_2d_spike_rate(img2, fig2,np.array(s21),'S2')
        # update_2d_spike_rate(img3, fig3,np.array(s22),'S2')
        # update_2d_spike_rate(img4, fig4,np.array(s23),'S2')
        # update_2d_spike_rate(img5, fig5,np.array(s24),'S2')

        cur_count1 = np.array(sc1) - pre_count1 # S1Ae的脉冲数
        cur_count2 = np.array(sc2) - pre_count2 # S1Be
        cur_count3 = np.array(sc3) - pre_count3
        cur_count4 = np.array(sc4) - pre_count4

        cur_count21 = np.array(scc1) - pre_count21

        pre_count1 = np.array(sc1)
        pre_count21 = np.array(scc1)

        save_2d_spike_rate(cur_count1, 'S1') # 把S1Ae的发放情况画出来并保存
        fig_num += 1
        save_2d_spike_rate(cur_count2, 'S1') # 把S1Be的发放情况画出来并保存
        fig_num += 1
        save_2d_spike_rate(cur_count3, 'S1')
        fig_num += 1
        save_2d_spike_rate(cur_count4, 'S1')
        fig_num += 1
        # save_2d_spike_rate(cur_count21, 'C1')
        # fig_num += 1

        # 画S2的结果
        # for name in S2_population_names: # ABCDEFGH
        #     cur_count['S2'+name+'e'] = spike_monitors['S2'+name+'e'].count[:] - pre_count['S2'+name+'e']
        #     cur_count['C2' + name + 'e'] = spike_monitors['C2' + name + 'e'].count[:] - pre_count['C2' + name + 'e']
        #     pre_count['S2'+name+'e'] = spike_monitors['S2'+name+'e'].count[:]
        #     pre_count['C2' + name + 'e'] = spike_monitors['C2' + name + 'e'].count[:]
        #     save_2d_spike_rate(cur_count['S2'+name+'e'], 'S2')
        #     fig_num += 1
        #     save_2d_spike_rate(cur_count['C2' + name + 'e'], 'C2')
        #     fig_num += 1

        # save_2d_spike_rate(cur_count['S2Be'], 'S2')
        # fig_num += 1
        # save_2d_spike_rate(cur_count['S2Ce'], 'S2')
        # fig_num += 1
        # save_2d_spike_rate(cur_count['S2De'], 'S2')
        # fig_num += 1
        # save_2d_spike_rate(cur_count['S2Ee'], 'S2')
        # fig_num += 1
        # save_2d_spike_rate(cur_count['S2Fe'], 'S2')
        # fig_num += 1
        # save_2d_spike_rate(cur_count['S2Ge'], 'S2')
        # fig_num += 1
        # save_2d_spike_rate(cur_count['S2He'], 'S2')
        # fig_num += 1
        # plot_2d_spike_rate(cur_count1, 'S1')
        # fig_num += 1
        # plot_2d_spike_rate(cur_count2, 'C1')
        # fig_num += 1
        # plot_2d_spike_rate(cur_count['S2Ae'], 'S2')
        # fig_num += 1
        # plot_2d_spike_rate(cur_count['S2Be'], 'S2')
        # fig_num += 1

        # 保存S2特征 + weight sharing
        # sum1 = len(np.where(cur_count['S2Ae'] == 0)[0])
        # sum2 = len(np.where(cur_count['S2Be'] == 0)[0])
        # sum3 = len(np.where(cur_count['S2Ce'] == 0)[0])
        # sum4 = len(np.where(cur_count['S2De'] == 0)[0])
        # sum5 = len(np.where(cur_count['S2Ee'] == 0)[0])
        # sum6 = len(np.where(cur_count['S2Fe'] == 0)[0])
        # sum7 = len(np.where(cur_count['S2Ge'] == 0)[0])
        # sum8 = len(np.where(cur_count['S2He'] == 0)[0])
        # for name in S2_population_names:
        #     idx['S2'+name+'e'] = np.where(cur_count['S2'+name+'e']==max(cur_count['S2'+name+'e']))[0][0]
        # # 保存特征，作为训练分类器的输入
        # i = j % 4
        # if i < 2: # 脸部特征
        #     #print('face:',sum1,sum2,sum3,sum4)
        #     with open('results/2017.8.17/face_feature_data.txt', 'a') as f:
        #         f.write(str([sum1, sum2, sum3, sum4,sum5, sum6, sum7, sum8, idx['S2Ae'],idx['S2Be'],idx['S2Ce'],
        #                      idx['S2De'],idx['S2Ee'],idx['S2Fe'],idx['S2Ge'],idx['S2He']]) + '\n')
        # if i >= 2:
        #     #print('motor:',sum1,sum2,sum3,sum4)
        #     with open('results/2017.8.17/motor_feature_data.txt', 'a') as f:
        #         f.write(str([sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, idx['S2Ae'],idx['S2Be'],idx['S2Ce'],
        #                      idx['S2De'],idx['S2Ee'],idx['S2Fe'],idx['S2Ge'],idx['S2He']]) + '\n')
        #
        # print('share weights')
        # for name in C1_S2_conn_names:
        #     for connType in C1_S2_conn_types:  # 'ee'
        #         connName = 'C1' + name[0] + connType[0] + 'S2' + name[1] + connType[1]  # connName='C1AeS2Ae' 'C1BeS2Ae'
        #         #print('更新权重：' + connName)
        #         # idx = np.where(cur_count[connName[4:8]]==max(cur_count[connName[4:8]]))[0][0] # 读出第一个发放的神经元序号
        #         copy_weights = np.array( connections[connName].w[ idx[connName[4:8]] * S2_cov_patch_size:(idx[connName[4:8]] + 1)
        #                                                                                                  * S2_cov_patch_size]/nS)
        #         connections[connName].w =np.tile(copy_weights, S2_num)*nS
        #         copy_theta = S2_groups[connName[6:8]].theta[idx[connName[4:8]]]/mV
        #         S2_groups[connName[6:8]].theta = np.tile(copy_theta,S2_num)*mV
        #         # print('更新完毕：' + connName)
        # # 提取卷积层特征
        # # with open('train_motor_feature_data.txt', 'a') as f:
        # #     f.write(str(list(np.concatenate((np.array(scc1), np.array(scc2), np.array(scc3), np.array(scc4)))))+'\n')
        # # A=np.array(connections['C1AeS2Ae'].w)
        # # B = np.array(connections['C1AeS2Be'].w)
        j += 1

training_end = time.time()
print("train time:", training_end - training_start)
# 所有num_examples个样本训练完后，保存结果
# ------------------------------------------------------------------------------
# save results
# ------------------------------------------------------------------------------
# print ('save results')
# if not test_mode:
#     save_theta()
# if not test_mode:
#     save_connections()
# else:
#     np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)  # 将每一个样本导致的各个神经元发放率保存下来
#     np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)

# ------------------------------------------------------------------------------
# plot results
# ------------------------------------------------------------------------------
# if rate_monitors:
#     b2.figure(fig_num)
#     fig_num += 1
#     for i, name in enumerate(rate_monitors):
#         b2.subplot(len(rate_monitors), 1, i + 1)
#         b2.plot(rate_monitors[name].t / second, rate_monitors[name].rate, '.')
#         b2.title('Rates of population ' + name)

# if spike_monitors:
#     b2.figure(fig_num)
#     fig_num += 1
#     for i, name in enumerate(spike_monitors):
#         b2.subplot(len(spike_monitors), 1, i + 1)
#         b2.plot(spike_monitors[name].t/ms,spike_monitors[name].i,'.')
#         b2.title('Spikes of population ' + name)
#
# if spike_monitors:
#     b2.figure(fig_num)
#     fig_num += 1
#     for i, name in enumerate(spike_monitors):
#         b2.subplot(len(spike_monitors), 1, i + 1)
#         b2.plot(spike_monitors['Ae'].count[:])
#         b2.title('Spike count of population ' + name)
# figure('AeV')
# plot(stateS2Ae.t/ms,stateS2Ae.v[0]/mV)
# figure('AeI')
# plot(stateS2Ae.t/ms,stateS2Ae.I_synI[0]/pamp)
# figure('AiV')
# plot(stateS2Ai.t/ms,stateS2Ai.v[0]/mV)
# figure('AiI')
# plot(stateS2Ai.t/ms,stateS2Ai.I_synE[0]/pamp)
# count_thr = 50  # 在运行时间内的脉冲发放阈值，与运行时间有关
# aaa = np.array(connections['C1AeS2Ae'].w[:])
# bbb = np.array(connections['C1BeS2Ae'].w[:])
# ccc = np.array(connections['C1AeS2Be'].w[:])
# print(sc1,scp1,scc1)
sc = sc1 + sc2 + sc3 + sc4
scc = scc1 + scc2 + scc3 + scc4
scp = scp1 + scp2 + scp3 + scp4

# plot_2d_spike_rate(np.array(sc), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scp), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scc), 'C1')
# fig_num += 1
#
# plot_2d_spike_rate(np.array(sc1), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scp1), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scc1), 'C1')
# fig_num += 1
#
# plot_2d_spike_rate(np.array(sc2), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scp2), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scc2), 'C1')
# fig_num += 1
#
# plot_2d_spike_rate(np.array(sc3), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scp3), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scc3), 'C1')
# fig_num += 1
#
# plot_2d_spike_rate(np.array(sc4), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scp4), 'S1')
# fig_num += 1
# plot_2d_spike_rate(np.array(scc4), 'C1')
# fig_num += 1
#
# plot_2d_spike_rate(np.array(s21), 'S2')
# fig_num += 1
# plot_2d_spike_rate(np.array(s22), 'S2')
# fig_num += 1
# plot_2d_spike_rate(np.array(s23), 'S2')
# fig_num += 1
# plot_2d_spike_rate(np.array(s24), 'S2')
# fig_num += 1
# plot_2d_input_weights()  # 权值可视化
# b2.ioff()
# b2.show()
