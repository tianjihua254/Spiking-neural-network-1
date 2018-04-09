# -*- coding:utf-8 -*-
import cv2 as cv2
from function import *

# parameters list
taum = 10 * ms  # time constant for membrane potential
taupre = 20 * ms  # time constant for Apre in STDP
taupost = 1.0 * taupre
Ee = 0 * mV  # 兴奋性逆转电势（不用修改）
Ei = -85 * mV  # 抑制性逆转电势（不用修改）
vt = -55 * mV  # 膜电位阈值
v_rest_e = -65 * mV  # 静息电位
v_rest_i = -60 * mV
v_reset_e = -65 * mV # 恢复电位
v_reset_i = -45*mV
v_thresh_e = -52. * mV  # 阈值
v_thresh_i = -40. * mV
taue = 1 * ms  # 兴奋性电导衰减的时间常数 5
taui = 2 * ms  # 抑制性电导衰减的时间常数 5
gleak = 1 * nS  # 漏电导（不用修改）！！！！！待定 10ns？1ns？
gemax = 4.0* nS  # 兴奋性电导的最大值
gimax = 1.5* nS
gmax = {}
gmax['e'] = gemax
gmax['i'] = gimax
I = 0 * pA  # （不用修改）

# dApre_e = .008
dApre_e = .01
dApre_e *= gemax  # 兴奋性突触权值改变量
dApost_e = dApre_e * taupre / taupost * 1.05

dApre_i = .008
dApre_i *= gimax  # 抑制性突触权值改变量
dApost_i = dApre_e * taupre / taupost * 1.05

# Neuron model
eqs_input = '''
dv/dt = ((v_reset_e-v)+(I+Ie+Ii)/gleak)/taum : volt (unless refractory)
Ie = ge*(Ee-v) : amp
Ii = gi*(Ei-v) : amp
ge : siemens
gi : siemens'''
poisson_neurons = '''
rates : Hz
dv/dt = 1 : second'''
eqs_neurons = '''
dv/dt = ((v_reset_e-v) + ( I + Ie + Ii)/gleak)/taum : volt (unless refractory)
Ie = ge*(Ee-v) : amp
Ii = gi*(Ei-v) : amp
dge/dt = -ge/taue :siemens
dgi/dt = -gi/taui :siemens
'''
# Synapse model and STDP rule
wmod = '''
    w:siemens
    dApre/dt = -Apre / taupre : siemens (event-driven)
    dApost/dt = -Apost / taupost : siemens (event-driven)
    '''
won_pre_e = '''
    ge+=w
    Apre+=dApre_e
    w = clip(w - Apost, 0, gemax)
    '''
won_post_e = '''
    Apost+=dApost_e
    w = clip(w + Apre, 0, gemax)
    '''
won_pre_i = '''
    gi+=w
    Apre+=dApre_i
    w = clip(w - Apost, 0, gimax)
    '''
won_post_i = '''
    Apost+=dApost_i
    w = clip(w + Apre, 0, gimax)
    '''
# 设置图像尺寸并且读入图像
imgsize = 200
length = imgsize**2
imgpath='images/airplanes/image_0700.jpg'
# imgpath='edge_test.png'
img = cv2.imread(imgpath,0)
cv2.imshow(" ",img)
# # cv2.waitKey(0)
img=cv2.resize(img,(imgsize,imgsize))

# img = np.ones((imgrows,imgcols))*255
# cv2.normalize(img,img, 1, 0, cv2.NORM_MINMAX)
# cv2.imshow("lena",img)

layers={} # 使用字典容器存放每一层的信息
layers_name=['Retina','LGN']
# 将图像像素对应为脉冲频率
max_rate = 30*Hz
rate = np.array((img*max_rate/255.0))
rate = reshape(rate,length)
layers['Retina'] = NeuronGroup(length, model=poisson_neurons, threshold='v>1/rates',reset='v=0*second',method='euler')
layers['Retina'].rates = rate*Hz

layers['LGN'] = NeuronGroup(length, model=eqs_neurons, method='euler', threshold='v>v_thresh_e', reset='v=v_reset_e', refractory=3 * ms)
layers['LGN'].v = -70 * mV
layers['LGN'].ge = 0 * nS
layers['LGN'].gi = 0 * nS

# 计算同心圆感受野连接范围
pre_on, post_on, pre_off,post_off = create_connmatrix(imgsize,1,5 ) # 感受野大小可调，最好为奇数

# 感受野ON区和OFF区的初始连接权要根据图片调整，不同初始值有不同效果
print("------正在连接------")
# 中心ON区
# S_on = Synapses(layers['Retina'],layers['LGN'], model=wmod, on_pre=won_pre_e, on_post=won_post_e, method='euler')
S_on = Synapses(layers['Retina'],layers['LGN'], model='w:siemens', on_pre='ge+=w', method='euler')
S_on.connect(i=pre_on, j=post_on)
S_on.w = 4.0*nS# 5.0*nS
print("ON区共有%d连接"%len(pre_on))

# 周围OFF区
# S_off = Synapses(layers['Retina'],layers['LGN'], model=wmod, on_pre=won_pre_i, on_post=won_post_i, method='euler')
S_off = Synapses(layers['Retina'],layers['LGN'], model='w:siemens', on_pre='gi+=w' , method='euler')
S_off.connect(i=pre_off, j=post_off)
S_off.w = 0.35*nS # <- face \\\\\ simple edge 0.3*nS
print("OFF区共有%d连接"%len(pre_off))
print("------连接完毕------")

# 将调试信息保存
# f=open('test_result/Retina_LGN.txt','a')
# # 保存结果的名称
# filename='test_result/edge_4.jpg'
# f.write('\n对应图像：%s'%filename)
# f.write('\nON区感受野连接权：%s'%S_on.w[0])
# f.write('\nOFF区感受野连接权：%s\n'%S_off.w[0])
# f.close()

# set the monitor
M1 = SpikeMonitor(layers['Retina'])
M2 = SpikeMonitor(layers['LGN'])

# run the simulation
net = Network(collect())
net.add(layers)
duration = 100*ms
net.run(duration, report='text')
print(M2.t,M2.i)
print(len(M2.t),len(M2.i))
temp_rate = M2.count/duration
out_pixel = []

# 将脉冲发放率转换为像素0-255
for r in temp_rate:
    r= r*(255.0/np.max(temp_rate))
    out_pixel.append(r)
out_pixel = np.array(out_pixel)
out_pixel = np.reshape(out_pixel,(imgsize,imgsize))
# 保存图像
# cv2.imwrite(filename,out_pixel)
# 归一化
cv2.normalize(out_pixel,out_pixel, 1, 0, cv2.NORM_MINMAX)
cv2.imshow("LGN_img",out_pixel)
# out_pixel=out_pixel*255
# cv2.imwrite("111motor.jpg",out_pixel)
cv2.waitKey(0)
show()
