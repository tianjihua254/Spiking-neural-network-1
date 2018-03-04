# -*- coding:utf-8 -*-
"""
Created on 17.7.2017
RF of V1 training using unsupervised STDP rule
@author: Jiaxing Liu
"""

import brian2 as b2
from brian2 import *
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import time
import os.path
import scipy
# import cPickle as pickle  # 对python对象序列化的保存和恢复
# import brian_no_units  # import it to deactivate unit checking --> This should NOT be done for testing/debugging
# import brian as b
from struct import unpack # 解压缩 将C语言格式数据转换成Python格式数据
# from brian import *
from glob import glob

# specify the location of the MNIST data
MNIST_data_path = ''


# ------------------------------------------------------------------------------
# functions`
# ------------------------------------------------------------------------------
def get_data(data_path): # 加载训练方向条
    imgdata = glob(data_path + '*.jpg')
    imgs = []
    for i in range(len(imgdata)):
        temp_img = b2.imread(imgdata[i])
        imgs.append(temp_img)
    return imgs

def get_matrix_from_file(fileName):  # 加载权值矩阵，初始化的权值矩阵以文件形式保存，需要的时候load进来
    offset = len(ending) + 4
    if fileName[-4 - offset] == 'X':  # 是否为输入，设置第一层个数 确定source和target层的神经元数量
        n_src = n_input
    else:
        if fileName[-3 - offset] == 'e':  # 兴奋性
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1 - offset] == 'e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print (readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))  # 全为0的权值矩阵
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:, 2]  # 将不为0的权值填充进去
    return value_arr

def save_connections(ending=''):  # 保存输入到兴奋层的权值
    print ('save connections')
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # date = '2017.7.26'
    for connName in save_conns:  # 'XeAe'
        connMatrix = np.array(connections[connName].w/nS).reshape((n_input, n_e))
        #         connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
        connListSparse = (
        [(i, j, connMatrix[i, j]) for i in range(connMatrix.shape[0]) for j in range(connMatrix.shape[1])])
        np.save(data_path + 'weights/' + connName + '_' + date + ending, connListSparse)

def save_theta(ending=''):  # 将兴奋层的神经元阈值参数theta保存下来
    print ('save theta')
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    for pop_name in population_names:  # 'A'
        np.save(data_path + 'weights/theta_' + pop_name + '_' + date + ending, neuron_groups[pop_name + 'e'].theta/mV)

def normalize_weights():  # 权值归一化，用每一列的权值总和来归一化每一个权值
    # 类似于感受野内连接之间的侧抑制
    for connName in save_conns:  # XeAe XeBe XeCe XeDe
        if connName[1] == 'e' and connName[3] == 'e':  # 兴奋到兴奋的连接，同个感受野内连接之间侧抑制
            connection = np.array(connections[connName].w/nS).reshape((n_input, n_e))
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis=0)  # 权值矩阵按列求和，即每个神经元感受野权值相加
            colFactors = weight['ee_input'] / colSums  # 期望总权值除以每一列的和
            for j in range(n_e):  # n_e=25，每个兴奋性神经元与输入15*15神经元相连
                connection[:, j] *= colFactors[j]
                #np.array(connections[connName].w / nS).reshape((n_input, n_e))[:,j]*=colFactors[j]
            connections[connName].w = connection.reshape(n_input*n_e)[:]*nS

def get_2d_input_weights():  # 将输入到兴奋层的连接转换为二维矩阵
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt * n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights={}
    twoD_weights = np.zeros((2 * num_values_col, 2 * num_values_row))
    for name in save_conns:
        rearranged_weights[name] = np.zeros((num_values_col, num_values_row))  # 105*105,将第二层每个兴奋性神经元的感受野二维排列出来
        connMatrix = np.array(connections[name].w / nS)
        weight_matrix = np.copy(connMatrix.reshape((n_input, n_e)))
        for i in range(n_e_sqrt):
            for j in range(n_e_sqrt):  # 每个神经元感受野权值赋值
                rearranged_weights[name][i * n_in_sqrt: (i + 1) * n_in_sqrt, j * n_in_sqrt: (j + 1) * n_in_sqrt] = \
                    weight_matrix[:, i + j * n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    # name = ['XeAe','XeBe','XeCe','XeDe']  # 输入层到第二层的连接为XeAe
    # weight_matrix = np.zeros((n_input, n_e))  # 15*15 X 5*5
    for i in range(len(rearranged_weights)/2):
        for j in range(len(rearranged_weights)/2):
            #a= rearranged_weights[save_conns[i * 2 + j]]
            twoD_weights[i*num_values_col:(i+1)*num_values_col,j*num_values_row:(j+1)*num_values_row] = rearranged_weights[save_conns[i*2+j]]
    return twoD_weights # 210*210

def plot_2d_input_weights():  # 将权重可视化出来
    name = 'XeAe'
    weights = get_2d_input_weights()
    # weights = np.ones((75, 75))
    fig = b2.figure(fig_num, figsize=(5, 5))
    im2 = b2.imshow(weights, interpolation="nearest", vmin=0, vmax=wmax_ee/nS, cmap=cmap.get_cmap('hot'))
    b2.colorbar(im2)  # 转换成伪彩图显示
    b2.title('weights of connection' + 'input-Ex')
    fig.canvas.draw()
    return im2, fig

def update_2d_input_weights(im, fig):  # 返回更新后的权值图案
    weights = get_2d_input_weights()  # 二维权值矩阵 15*5 X 15*5
    im.set_array(weights)
    b2.pause(0.005)
    fig.canvas.draw()
    return im

def sparsenMatrix(baseMatrix, pConn): # 将一个权值矩阵变为稀疏的矩阵
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    while numWeights < int(numTargetWeights):
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList

def generate_weughts():
    nInput = 5 * 5  # 5*5
    nE = 10 * 10
    nI = nE
    dataPath = './S1_random_weights/'
    weight = {}
    weight['ee_input'] = 0.85  # 0.3
    weight['ei_input'] = 0.2
    weight['ee'] = 0.1
    weight['ei'] = 10.5  # 兴奋到抑制连接的权值
    weight['ie'] = 12.5
    weight['ii'] = 0.4
    pConn = {}
    pConn['ee_input'] = 1.0
    pConn['ei_input'] = 0.1
    pConn['ee'] = 1.0
    pConn['ei'] = 0.0025
    pConn['ie'] = 0.9
    pConn['ii'] = 0.1

    print 'create random connection matrices from E->E'
    connNameList = ['XeAe', 'XeBe', 'XeCe', 'XeDe']
    for name in connNameList:
        randlist = 1.2 * np.random.random(nInput) + 0.05
        weightMatrix = (np.ones((nE, nInput)) * randlist).transpose()
        # weightMatrix = np.random.random((nInput, nE)) + 0.01
        weightMatrix *= weight['ee_input']
        weightList = [(i, j, weightMatrix[i, j]) for j in range(nE) for i in range(nInput)]
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)

    print 'create connection matrices from E->I which are purely random'
    connNameList = ['AeAi', 'BeBi', 'CeCi', 'DeDi']
    for name in connNameList:
        if nE == nI:
            weightList = [(i, i, weight['ei']) for i in range(nE)]
        else:
            weightMatrix = np.random.random((nE, nI))
            weightMatrix *= weight['ei']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)

    print 'create connection matrices from I->E which are purely random'
    connNameList = ['AiBe', 'AiCe', 'AiDe', 'BiAe', 'BiCe', 'BiDe', 'CiAe', 'CiBe', 'CiDe', 'DiAe', 'DiBe', 'DiCe']
    for name in connNameList:
        if nE == nI:
            weightList = [(i, i, weight['ie']) for i in range(nI)]
            # weightMatrix = np.ones((nI, nE))
            # weightMatrix *= weight['ie']
            # for i in xrange(nI):
            #     weightMatrix[i,i] = 0
            # weightList = [(i, j, weightMatrix[i,j]) for i in xrange(nI) for j in xrange(nE)]
        else:
            weightMatrix = np.random.random((nI, nE))
            weightMatrix *= weight['ie']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        print 'save connection matrix', name
        np.save(dataPath + name, weightList)


# def get_current_performance(performance, current_example_num):  # 当前性能，每10000个训练样本算一次精度
#     current_evaluation = int(current_example_num / update_interval)
#     start_num = current_example_num - update_interval
#     end_num = current_example_num
#     difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
#     correct = len(np.where(difference == 0)[0])
#     performance[current_evaluation] = correct / float(update_interval) * 100
#     return performance
#
#
# def plot_performance(fig_num):
#     num_evaluations = int(num_examples / update_interval)  # 有多少组精度值
#     time_steps = range(0, num_evaluations)
#     performance = np.zeros(num_evaluations)
#     fig = b.figure(fig_num, figsize=(5, 5))
#     fig_num += 1
#     ax = fig.add_subplot(111)
#     im2 = ax.plot(time_steps, performance)
#     b.ylim(ymax=100)
#     b.title('Classification performance')
#     fig.canvas.draw()
#     return im2, performance, fig_num, fig
#
#
# def update_performance_plot(im, performance, current_example_num, fig):  # 将新一组的精度值画出来
#     performance = get_current_performance(performance, current_example_num)
#     im.set_ydata(performance)
#     fig.canvas.draw()
#     return im, performance
#
#
# def get_recognized_number_ranking(assignments, spike_rates):  # 识别数字的排序
#     summed_rates = [0] * 10
#     num_assignments = [0] * 10
#     for i in range(10):
#         num_assignments[i] = len(np.where(assignments == i)[0])
#         if num_assignments[i] > 0:
#             summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
#     return np.argsort(summed_rates)[::-1]  # 按照神经元发放率给数字0-9排序
#
#
# def get_new_assignments(result_monitor, input_numbers):  # 给兴奋性神经元分配类别，20*20=400个神经元每个都属于0-9其中一个
#     assignments = np.zeros(n_e)  # 20*20
#     input_nums = np.asarray(input_numbers)  #
#     maximum_rate = [0] * n_e  # 20*20
#     for j in range(10):
#         num_assignments = len(np.where(input_nums == j)[0])  # 输入为j的个数
#         if num_assignments > 0:
#             rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments
#         for i in range(n_e):
#             if rate[i] > maximum_rate[i]:
#                 maximum_rate[i] = rate[i]
#                 assignments[i] = j
#     return assignments


# ------------------------------------------------------------------------------
# load orientation
# ------------------------------------------------------------------------------
start = time.time()
img=get_data('orientation bar/')
#training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print('time needed to load training set:', end - start)

# start = time.time()
# testing = get_labeled_data(MNIST_data_path + 'testing', bTrain=False)
# end = time.time()
# print 'time needed to load test set:', end - start

# ------------------------------------------------------------------------------
# set parameters and equations
# ------------------------------------------------------------------------------
test_mode = False

# 很多设置在linux下可用，windows下会报错
# b2.set_global_preferences(
#     defaultclock=b.Clock(dt=0.5 * b.ms),
#     # The default clock to use if none is provided or defined in any enclosing scope.
#     useweave=False,  # Defines whether or not functions should use inlined compiled C code where defined.
#     gcc_options=['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler.
#     # For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on
#     usecodegen=False,  # Whether or not to use experimental code generation support.
#     usecodegenweave=False,  # Whether or not to use C with experimental code generation support.
#     usecodegenstateupdate=False,  # Whether or not to use experimental code generation support on state updaters.
#     usecodegenthreshold=False,  # Whether or not to use experimental code generation support on thresholds.
#     usenewpropagate=False,  # Whether or not to use experimental new C propagation functions.
#     usecstdp=False,  # Whether or not to use experimental new C STDP.
# )

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
    # weight_path = data_path + 'random/'
    weight_path = './S1_random_weights/'
    num_examples = 700 * 1  # 训练样本集60000重复训练3次
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 60000:
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True  # 输入到兴奋层通过STDP学习

ending = '' # 路径结尾的名称
n_input =25   # 5*5
# 兴奋层有四张网络，分别对应四个方向，此处暂时设置一个方向
n_e = 10*10
n_i = n_e  # 兴奋性神经元与抑制性神经元一对一连接
single_example_time = 0.35 * second  # 每个样本训练时间为350ms，然后切换下一张图像
resting_time = 0.15 * second  # 间隔150ms，使与时间有关的参数回归初始值，例如电导ge、内稳态变量theta等
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:
    update_interval = num_examples
    weight_update_interval = 20
else:  # 训练
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

v_rest_e = -65. * mV  # 静息电位
v_rest_i = -60. * mV
v_reset_e = -65. * mV  # 恢复电位
v_reset_i = -45. * mV
v_thresh_e = -52. * mV  # 兴奋性阈值
v_thresh_i = -40. * mV
refrac_e = 5. * ms  # 不应期，抑制性神经元的发放频率高，所以不应期短，时间常数小
refrac_i = 2. * ms
gleak = 1. * nS
taum_e = 100. * ms
taum_i = 10*ms # 抑制性神经元时间常数小，发放快
# gemax = 2.0* nS  # 兴奋性电导的最大值
# gimax = 1.5* nS

conn_structure = 'dense'  # 连接方式为全连接
weight = {}  # 以字典格式存储权值
delay = {}
input_population_names = ['X']  # 输入层名字
population_names = ['A','B','C','D']  # 兴奋层 应该有四个 A，B，C，D
input_connection_names = ['XA','XB','XC','XD']  # 输入到兴奋的连接
save_conns = ['XeAe','XeBe','XeCe','XeDe']  # 保存输入到兴奋的连接
input_conn_names = ['ee_input']  # 输入到兴奋的连接为ee
recurrent_conn_names = ['ei', 'ie']  # 兴奋到抑制的连接
weight['ee_input'] = 50.05 # 随着输入需要改变，与输入图像的占空比有关
# weight['ee_input'] = 40 # 随着输入需要改变，与输入图像的占空比有关
# weight['ee_input'] = {'45':35,'135':35,'Ver':15,'Hor':15}  # 权值归一化时，每一列的期望值，权值之间竞争的期望总和
delay['ee_input'] = (0 * ms, 10 * ms)
delay['ei_input'] = (0 * ms, 5 * ms)
input_intensity = 2.  # 输入强度，像素值向脉冲频率转换的强度
start_input_intensity = input_intensity

tc_pre_ee = 20 * ms  # STDP时间常数
tc_post_ee = 20 * ms # 突触后变量衰减慢一些
nu_ee_pre = 0.0002 # 衰减learning rate 决定了感受野其他区域能否被抑制为0
nu_ee_post = 0.015  # 增加learning rate 决定了学习出感受野的速度
wmax_ee = 10 * nS  # 权值最大值
wmax_ei = 3.5 * nS
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre = 0.2
STDP_offset = 0.4
w_mu_pre = 0.2
w_mu_post = 0.2

# 设置发放阈值
if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 5e6 * ms  # 阈值参数theta衰减的很慢
    theta_plus_e = 0.02 * mV  # 每发放一个脉冲，阈值增加plus_e 阈值增加的快慢
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'  # 每发放一次，theta就改变一次
offset = 20.0 * mV

# 兴奋性神经元膜电位时间常数为100ms 兴奋性电导时间常数为1ms
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / gleak) / taum_e  : volt
        I_synE = ge *         -v                                : amp
        I_synI = gi * (-100.*mV-v)                              : amp
        dge/dt = -ge/(1.0*ms)                                   : siemens
        dgi/dt = -gi/(2.0*ms)                                   : siemens
      '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 100.0  : second'

# 抑制性神经元的时间常数小，发放快
neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / gleak) / taum_i  : volt
        I_synE = ge *         -v                                : amp
        I_synI = gi * (-85.*mV-v)                               : amp
        dge/dt = -ge/(1.0*ms)                                   : siemens
        dgi/dt = -gi/(2.0*ms)                                   : siemens
      '''

# determine STDP rule to use，确定使用哪种STDP规则
stdp_input = ''

if raw_input('Use weight dependence (default no)?: ') in ['no', '']:
    use_weight_dependence = False
    stdp_input += 'no_weight_dependence_'
else:
    use_weight_dependence = True
    stdp_input += 'weight_dependence_'

if raw_input('Enter (yes / no) for post-pre (default yes): ') in ['yes', '']:
    post_pre = True
    stdp_input += 'postpre'
else:
    post_pre = False
    stdp_input += 'no_postpre'

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
# Brian2用的STDP学习规则，类似于不依赖当前权重的STDP
# wmod = '''
#     w:siemens
#     dApre/dt = -Apre / taupre : siemens (event-driven)
#     dApost/dt = -Apost / taupost : siemens (event-driven)
#     '''
# won_pre_e = '''
#     ge+=w
#     Apre+=dApre_e
#     w = clip(w - Apost, 0, gemax)
#     '''
# won_post_e = '''
#     Apost+=dApost_e
#     w = clip(w + Apre, 0, gemax)
#     '''
# won_pre_i = '''
#     gi+=w
#     Apre+=dApre_i
#     w = clip(w - Apost, 0, gimax)
#     '''
# won_post_i = '''
#     Apost+=dApost_i
#     w = clip(w + Apre, 0, gimax)
#     '''

b2.ion()
fig_num = 1
neuron_groups = {}  # 神经元群
input_groups = {}  # 输入神经元群
connections = {}  # 连接群
rate_monitors = {} # 发放频率监视器
spike_monitors = {} # 脉冲个数、脉冲发放时间、索引监视器
# result_monitor = np.zeros((update_interval, n_e))  # 10000*25，每个神经元对样本的反应
# 兴奋层有四组神经元，对应四个方向功能柱，15*15*4=900
neuron_groups['e'] = b2.NeuronGroup(n_e * len(population_names), model=neuron_eqs_e, threshold='v>(theta - offset - 52*mV)',refractory=refrac_e,
                                   reset=scr_e, method='euler')
neuron_groups['i'] = b2.NeuronGroup(n_i * len(population_names), model=neuron_eqs_i, threshold='v>v_thresh_i', refractory=refrac_i,
                                   reset='v=v_reset_i', method='euler')
# 兴奋抑制连接
# ------------------------------------------------------------------------------
# create network population and recurrent connections
# ------------------------------------------------------------------------------
generate_weughts()
for idx,name in enumerate(population_names):  # 创建若干神经元子群 'A' 'B' 'C' 'D'
    print('create neuron group', name)

    neuron_groups[name + 'e'] = neuron_groups['e'][idx*n_e:(idx+1)*n_e]  # neuron_groups['Ae']
    neuron_groups[name + 'i'] = neuron_groups['i'][idx*n_i:(idx+1)*n_i]  # neuron_groups['Ai']

    neuron_groups[name + 'e'].v = v_rest_e - 10. * mV
    neuron_groups[name + 'i'].v = v_rest_i - 10. * mV
    neuron_groups[name + 'e'].ge = 0*nS
    neuron_groups[name + 'i'].gi = 0*nS
    if test_mode: #or weight_path[-8:] == 'weights/':  # 加载阈值参数值
        neuron_groups[name + 'e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy')
    else:
        neuron_groups[name + 'e'].theta = np.ones((n_e)) * 20.0 * mV  # 训练时，theta的初始值为20mV


    print ('create recurrent connections')  # 建立兴奋与抑制的连接
    conn_type = recurrent_conn_names[0]  # 'ei'
    connName = name + conn_type[0] + name + conn_type[1]  # AeAi AiAe BeBi BiBe
    weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')  # 初始化权值
    if(conn_type=='ei'):
        connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], model='w:siemens',
                                            on_pre='ge+=w',method='euler')# conn_type='ei'/'ie'
        connections[connName].connect(j='i')  # 建立一对一连接
        weightlist = weightMatrix.reshape((n_e * n_e))
        connections[connName].w = weightlist[np.where(weightlist != 0)[0]]*nS

# 建立抑制连接，每个抑制性神经元与同一位置的其他三种神经元连接
for name in population_names:  # A B C D
    for name2 in population_names:
        conn_type = recurrent_conn_names[1] # 'ie'
        if name2==name:
            pass
        else:
            connName = name + conn_type[0] +name2 +conn_type[1]
            weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy') # 加载初始权值
            if (conn_type == 'ie'):
                connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                        model='w:siemens', on_pre='gi+=w', method='euler')  # conn_type='ie'
                connections[connName].connect(j='i')  # 建立一对一连接
                weightlist = weightMatrix.reshape((n_e * n_e))
                connections[connName].w = weightlist[np.where(weightlist != 0)[0]]*nS

    print ('create monitors for', name)  # monitors for A
    rate_monitors[name + 'e'] = b2.PopulationRateMonitor(neuron_groups[name + 'e']) # 350ms+150ms
    rate_monitors[name + 'i'] = b2.PopulationRateMonitor(neuron_groups[name + 'i'])

    if record_spikes:
        spike_monitors[name + 'e'] = b2.SpikeMonitor(neuron_groups[name + 'e'])
        spike_monitors[name + 'i'] = b2.SpikeMonitor(neuron_groups[name + 'i'])

# if record_spikes:
#     b2.figure(fig_num)  # 创建一个figure对象,显示监测到的脉冲散点图
#     fig_num += 1
#     b2.ion()
#     b2.subplot(111)
#     b2.plot(spike_monitors['Ae'].t / ms, spike_monitors['Ae'].i, '.')  # 画出脉冲发放的散点图
#     b2.title('Spike by Time of Ae')

    # b2.subplot(212)
    # b2.plot(spike_monitors['Ai'].t / ms, spike_monitors['Ai'].i, '.', refresh=1000 * ms, showlast=1000 * ms)
    # b2.title('Spike by Time of Ai')

# ------------------------------------------------------------------------------
# create input population and connections from input populations
# ------------------------------------------------------------------------------
pop_values = ['A', 'B', 'C','D']  # 创建输入到兴奋层的连接
for i, name in enumerate(input_population_names):  # 'X'
    input_groups[name + 'e'] = b2.PoissonGroup(n_input, 0*Hz)  # input_groups['Xe'] n_input=15*15
    rate_monitors[name + 'e'] = b2.PopulationRateMonitor(input_groups[name + 'e'])

for name in input_connection_names:  # 'XA' 'XB' 'XC' 'XD'
    print ('create connections between', name[0], 'and', name[1])
    for connType in input_conn_names:  # 'ee'
        connName = name[0] + connType[0] + name[1] + connType[1]  # connName='XeAe'
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')  # 加载初始权值矩阵
        connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]], model=eqs_stdp_ee,
                                            on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee, method='euler')
        connections[connName].connect() # 全连接
        connections[connName].delay = 0*ms # delay[connType]  # 突触延迟是一个范围，0-10之间
        connections[connName].w = weightMatrix.reshape((n_input*n_e))*nS # weightMatrix.reshape((49*225))*siemens

# print(connections['XeAe'].w) # 初始的第一层权值
# print(connections['XeBe'].w) # 初始的第一层权值
# print(connections['XeCe'].w) # 初始的第一层权值
# print(connections['XeDe'].w) # 初始的第一层权值

# ------------------------------------------------------------------------------
# run the simulation and set inputs
# ------------------------------------------------------------------------------
previous_spike_count = np.zeros(n_e)  # 25 记录每个神经元脉冲数量
# assignments = np.zeros(n_e)  # 400
# input_numbers = [0] * num_examples  # 60000
# outputNumbers = np.zeros((num_examples, 10))  # 60000*10，每个输入样本的预测结果都有10种可能
if not test_mode:  # 可视化训练的权值
    input_weight_monitor, fig_weights = plot_2d_input_weights() # 权值可视化，75*75的矩阵
    #b2.close(fig_num)
    fig_num += 1
# if do_plot_performance:  # 将精度画出来
#     performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
for i, name in enumerate(input_population_names):  # 'X'
     input_groups[name + 'e'].rates = 0*Hz

net = Network(collect())
net.add(neuron_groups)
net.add(input_groups)
net.add(connections)
net.add(spike_monitors)
net.run(0*ms)
j = 0
training_start = time.time()
# 多张图片写成一个连起来的输入刺激，节省时间
# rate = TimedArray(np.tile(np.concatenate((np.concatenate((img[0].reshape((n_input))*3, np.zeros(n_input)),axis=0)*10,
#                                   np.concatenate((img[1].reshape((n_input)) * 3, np.zeros(n_input)), axis=0) * 10,
#                                   np.concatenate((img[2].reshape((n_input)) * 3, np.zeros(n_input)), axis=0) * 10,
#                                   np.concatenate((img[3].reshape((n_input)) * 3, np.zeros(n_input)), axis=0) * 10),
#                                  axis=0),200).reshape((-1,n_input))*Hz, dt=100.*ms)
while j < (int(num_examples)):  # 大循环，每个训练样本跑一次
    # if test_mode:
    #     if use_testing_set:  # 读取输入并将其转换为1*28*28的向量
    #         rates = testing['x'][j % 10000, :, :].reshape((n_input)) / 8. * input_intensity  # input_intensity=2
    #     else:
    #         rates = training['x'][j % 60000, :, :].reshape((n_input)) / 8. * input_intensity
    # else:
    normalize_weights()  # 权值归一化，起到竞争、侧抑制的作用
    # rate = img[j%4].reshape((n_input)) / 8. * input_intensity
    # rates = img[j % 60000].reshape((n_input))/ 8. * input_intensity
    # i = j%80
    # if i<=20:
    #     rate = img[0].reshape((n_input)) / 8. * input_intensity
    # if(i>20 and i<=40):
    #     rate = img[1].reshape((n_input)) / 8. * input_intensity
    # if (i>40 and i<=60):
    #     rate = img[2].reshape((n_input)) / 8. * input_intensity
    # if (i>60):
    #     rate = img[3].reshape((n_input)) / 8. * input_intensity
    i = j % 40
    if i <= 10:
        rate = img[0].reshape((n_input)) / 8. * input_intensity
    if (i > 10 and i <= 20):
        rate = img[1].reshape((n_input)) / 8. * input_intensity
    if (i > 20 and i <= 30):
        rate = img[2].reshape((n_input)) / 8. * input_intensity
    if (i > 30):
        rate = img[3].reshape((n_input)) / 8. * input_intensity
    input_groups['Xe'].rates = rate*Hz
    print 'run number:', j+1, 'of', int(num_examples)
    net.run(single_example_time,report='text')  # 每个样本运行350ms

    # 运行完一个样本后，更新类别、更新权重、保存参数
    # if j % update_interval == 0 and j > 0:  # 每10000个样本更新一次标签分配
    #     assignments = get_new_assignments(result_monitor[:], input_numbers[j - update_interval: j])
    current_spike_count = np.asarray(spike_monitors['Ae'].count[:]+spike_monitors['Be'].count[:]+
                                     spike_monitors['Ce'].count[:]+spike_monitors['De'].count[:]) \
                          - previous_spike_count  # 记录Ae每个神经元发放的脉冲数
    if j % weight_update_interval == 0 and not test_mode:  # 每100个样本更新一次权重
        update_2d_input_weights(input_weight_monitor, fig_weights)
        print('weights < 0.2 :',len(np.where(connections['XeAe'].w / nS < 0.2)[0]))
        print(neuron_groups['Ae'].theta - offset - 52 * mV)
        print('spike count', current_spike_count)
    # if j % save_connections_interval == 0 and j > 0 and not test_mode:  # 间隔保存权值和theta参数
    #     save_connections(str(j))
    #     save_theta(str(j))

    previous_spike_count = np.copy(spike_monitors['Ae'].count[:]+spike_monitors['Be'].count[:]+
                                     spike_monitors['Ce'].count[:]+spike_monitors['De'].count[:])
    #print(connections['XeAe'].w/nS)
    if np.sum(current_spike_count) < 15:  # 如果Ae总共脉冲数小于5
        # update_2d_input_weights(input_weight_monitor, fig_weights)
        # lowweights = len(np.where(connections['XeAe'].w / nS < 0.2)[0])
        # print('low weights < 0.25 has:',lowweights)
        # print('spike count', current_spike_count)
        # print(neuron_groups['Ae'].ge)
        # print(neuron_groups['Ae'].theta - offset - 52*mV)
        input_intensity += 1  # 输入像素转换的脉冲频率大一倍

        for i, name in enumerate(input_population_names):  # 'X'
            input_groups[name + 'e'].rates = 0*Hz
        net.run(resting_time)  # 暂停输入，使参数静息
    else:
        # result_monitor[j % update_interval, :] = current_spike_count  # 将第j个样本引发的400个神经元频率记录

        # if test_mode and use_testing_set:
        #     input_numbers[j] = testing['y'][j % 10000][0]
        # else:
        #     input_numbers[j] = training['y'][j % 60000][0]  # 获取样本标签
        # outputNumbers[j, :] = get_recognized_number_ranking(assignments, result_monitor[j % update_interval, :])
        if j % 50 == 0 and j > 0:  # 每100个样本输出一个
            print ('runs done:', j, 'of', int(num_examples))
        # if j % update_interval == 0 and j > 0:
        #     if do_plot_performance:  # 每10000个样本计算一次精度
        #         unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
        #         print ('Classification performance',
        #                performance[:(j / float(update_interval)) + 1])  # performance的长度为num_examples/update_interval
        for i, name in enumerate(input_population_names):  # 'X'
            input_groups[name + 'e'].rates = 0*Hz  # 暂停输入
        net.run(resting_time)
        input_intensity = start_input_intensity
        j += 1
training_end = time.time()
print("train time:", training_end - training_start)
# 所有num_examples个样本训练完后，保存结果
# ------------------------------------------------------------------------------
# save results
# ------------------------------------------------------------------------------
print ('save results')
if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
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

if spike_monitors:
    b2.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b2.subplot(len(spike_monitors), 1, i + 1)
        b2.plot(spike_monitors[name].t/ms,spike_monitors[name].i,'.')
        b2.title('Spikes of population ' + name)

if spike_monitors:
    b2.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b2.subplot(len(spike_monitors), 1, i + 1)
        b2.plot(spike_monitors['Ae'].count[:])
        b2.title('Spike count of population ' + name)

plot_2d_input_weights()  # 权值可视化
b2.ioff()
b2.show()

