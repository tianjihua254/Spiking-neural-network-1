# -*- coding:utf-8 -*-
'''
Created on 2017.8.7

@author: Jiaxing Liu
'''
# 主要功能：层次网络的主要参数
import brian2 as b2
from brian2 import *

# 每一层的网络结构参数
input_size = 128
input_num = input_size**2
S1_cov_stride = 1 # V1感受野滑动的步长
S1_cov_padding = 'valid' # 卷积后大小是否减小
S1_cov_patch = 7 # RF size of V1 cells
S1_cov_patch_size = S1_cov_patch**2
S1_size = 122
S1_num = S1_size*S1_size

C1_pool_patch = 4
C1_pool_stride = 0
C1_size = 30
C1_num = C1_size*C1_size

S2_cov_stride = 1
S2_cov_padding = 'valid'
S2_cov_patch = 15 # S2层卷积窗口
S2_cov_patch_size = S2_cov_patch**2
S2_size = 16 # S2层网络大小
S2_num = S2_size**2

C2_pool_patch = 2
C2_pool_stride = 0
C2_size = 8
C2_num = C2_size*C2_size

# 神经元模型参数
v_rest_e = -65. * mV  # 静息电位
v_rest_i = -60. * mV
v_reset_e = -65. * mV  # 恢复电位
v_reset_i = -50. * mV #-55
v_thresh_e = -52. * mV  # 兴奋性阈值
v_thresh_i = -40. * mV #-45
refrac_e = 5. * ms  # 不应期，抑制性神经元的发放频率高，所以不应期短，时间常数小
refrac_i = 2. * ms
gleak = 1. * nS
taum_e = 100. * ms
taum_i = 20*ms # 抑制性神经元时间常数小，发放快


input_intensity = 2.  # 输入强度，像素值向脉冲频率转换的强度
start_input_intensity = input_intensity

# STDP学习超参数
tc_pre_ee = 20 * ms  # STDP时间常数
tc_post_ee = 20 * ms # 突触后变量衰减慢一些
nu_ee_pre = 0.0001 # 衰减learning rate 决定了感受野其他区域能否被抑制为0
nu_ee_post = 0.021  # 增加learning rate 决定了学习出感受野的速度
wmax_ee = 4.5 * nS  # 权值最大值
wmax_ei = 3.5 * nS
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre = 0.2
STDP_offset = 0.4
w_mu_pre = 0.2
w_mu_post = 0.2

# 网络中固定连接的权值
weight = {}  # 不需要STDP学习的层间连接权值
# weight['ee_input'] = 14.05 # 随着输入需要改变，与输入图像的占空比有关
# weight['ee_input'] = 40 # 随着输入需要改变，与输入图像的占空比有关，权值归一化时，每一列的期望值，权值之间竞争的期望总和
weight['pool1I_E'] = 20.0*nS
weight['pool1E_I'] = 10*nS
weight['pool1_C1'] = 25.7*nS
weight['S1_pool1'] = 25.8*nS # 15.0*nS
weight['S2E_I'] = 11.8*nS
weight['S2I_E'] = 15.8*nS # 36----6.0*nS
weight['S2IE_single'] = 18.5*nS
weight['S2_C2'] = 11.5*nS
weight['C1_S2_norm'] = 225

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

input_groups = {}  # 输入神经元群
S1_groups = {}  # v1神经元群
pool1_groups = {}
C1_groups = {}
S2_groups = {}
pool2_groups = {}
C2_groups = {}
connections = {}  # 连接群
rate_monitors = {}  # 发放频率监视器
spike_monitors = {}  # 脉冲个数、脉冲发放时间、索引监视器

