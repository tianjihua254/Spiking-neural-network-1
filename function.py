# -*- coding:utf-8 -*-
from brian2 import *
import numpy as np
import matplotlib.cm as cmap
import math

def create_connmatrix(dim, radius_on,radius_off): # tested @ 2017-1-14
    pre_mtx_on=[]
    post_mtx_on=[]
    pre_mtx_off = []
    post_mtx_off = []
    for row in np.arange(np.floor(radius_off/2),dim-np.floor(radius_off/2)): #row,col遍历图片的行、列，去除边缘
        for col in np.arange(np.floor(radius_off/2),dim-np.floor(radius_off/2)):

            for i in np.arange(row-np.floor(radius_on/2),row+np.floor(radius_on/2)+1): # i,j遍历ON感受野内部的行、列
                for j in np.arange(col-np.floor(radius_on/2),col+np.floor(radius_on/2)+1):
                    pre_mtx_on.append(int(i*dim+j))
                    post_mtx_on.append(int(row*dim+col))

            # i,j遍历OFF感受野内部的行、列
            for i in np.arange(row-np.floor(radius_off/2),row-np.floor(radius_on/2)):
                for j in np.arange(col-np.floor(radius_off/2),col+np.floor(radius_off/2)+1):
                    pre_mtx_off.append(int(i*dim+j))
                    post_mtx_off.append(int(row*dim+col))
            for i in np.arange(row - np.floor(radius_on / 2), row + np.floor(radius_on / 2)+1):  # i,j遍历OFF感受野内部的行、列
                for j in np.arange(col - np.floor(radius_off / 2), col - np.floor(radius_on / 2) ):
                    pre_mtx_off.append(int(i * dim + j))
                    post_mtx_off.append(int(row * dim + col))
                for j in np.arange(col + np.floor(radius_on / 2)+1, col + np.floor(radius_off / 2)+1 ):
                    pre_mtx_off.append(int(i * dim + j))
                    post_mtx_off.append(int(row * dim + col))
            for i in np.arange(row+np.floor(radius_on/2)+1,row+np.floor(radius_off/2)+1): # i,j遍历OFF感受野内部的行、列
                for j in np.arange(col-np.floor(radius_off/2),col+np.floor(radius_off/2)+1):
                    pre_mtx_off.append(int(i*dim+j))
                    post_mtx_off.append(int(row*dim+col))
    return pre_mtx_on,post_mtx_on,pre_mtx_off,post_mtx_off


# def create_connmatrix(dim_pre, dim_post, radius, p=1 ,selfconn=True):  # tested @ 2017-1-14
#     # create connection index,
#     #
#     dim_pre = int(dim_pre)
#     dim_post = int(dim_post)
#     matrix_pre = []   # pre-neurons index
#     matrix_post = []  # post-neurons index
#     core_weight = []  # weight-core, storing the gaussian values (according to the distance) of every single connection
#     scale = 1.0*dim_pre/dim_post
#     projectcoordinate = linspace(scale / 2 - 0.5, scale * (dim_post - 1) + scale / 2 - 0.5, dim_post, endpoint=True)
#     # evenly project post-neuron to pre-plane
#     if p > 1.0 or p < 0.0:
#         print('connection probability out of range [0,1]!!!')
#         raise ValueError
#     else:
#         for r in range(len(projectcoordinate)):  # (r,c) is the post-neuron coordinate at post-plane,
#             #  (row,col)is the coordinate of the position where post-neuron projected to pre-plane
#
#             for c in range(len(projectcoordinate)):
#                 row = projectcoordinate[r]
#                 col = projectcoordinate[c]
#                 # Connect neurons where 'distance <= radius' (neurons at the edge where distance=radius are included)
#                 #
#                 for i in range(np.clip(int(ceil(row - radius)), 0, dim_pre),  # (i,j) is the coordinate of pre-neuron
#                                 np.clip(int(floor(row + radius)) + 1, 0, dim_pre)):
#                     for j in range(np.clip(int(ceil(col - sqrt(radius ** 2 - (i - row) ** 2))), 0, dim_pre),
#                                     np.clip(int(floor(col + sqrt(radius ** 2 - (i - row) ** 2))) + 1, 0, dim_pre)):
#                         if np.random.rand() < p:
#                             matrix_pre.append(i * dim_pre + j)
#                             matrix_post.append(r * dim_post + c)
#                             distance =(row-i)**2+(col-j)**2
#                             core_weight.append((1/(sqrt(2*pi)*radius**2))*math.exp(-distance/(2*(radius**2))))
#                             #core_weight.append(math.exp(distance / (2 * (radius ** 2))))
#         if len(matrix_pre) == len(matrix_post) and len(core_weight) == len(matrix_post):
#             print('total connections = '+str(len(matrix_pre))+'\naverage = '+str(1. * len(matrix_pre) / dim_pre ** 2)+' per pre-neuron'+
#                   '\naverage = '+str(1. * len(matrix_pre) / dim_post ** 2)+' per post-neuron')
#         else:
#             print('connection index error!!! source dimension does not euqal to target dimension!!!')
#             raise Exception
#     return matrix_pre, matrix_post, np.array(core_weight)


# def create_connmatrix(dim_pre, dim_post, radius, p,selfconn=True):  # tested @ 2017-1-14
#     # create connection index,
#     #
#     dim_pre = int(dim_pre)
#     dim_post = int(dim_post)
#     matrix_pre = []   # pre-neurons index
#     matrix_post = []  # post-neurons index
#     core_weight = []  # weight-core, storing the gaussian values (according to the distance) of every single connection
#     scale = 1.0*dim_pre/dim_post
#     projectcoordinate = linspace(scale / 2 - 0.5, scale * (dim_post - 1) + scale / 2 - 0.5, dim_post, endpoint=True)
#     # evenly project post-neuron to pre-plane
#     if p > 1.0 or p < 0.0:
#         print('connection probability out of range [0,1]!!!')
#         raise ValueError
#     else:
#         for r in range(len(projectcoordinate)):  # (r,c) is the post-neuron coordinate at post-plane,
#             #  (row,col)is the coordinate of the position where post-neuron projected to pre-plane
#
#             for c in range(len(projectcoordinate)):
#                 row = projectcoordinate[r]
#                 col = projectcoordinate[c]
#                 # Connect neurons where 'distance <= radius' (neurons at the edge where distance=radius are included)
#                 #
#                 for i in range(np.clip(int(ceil(row - radius)), 0, dim_pre),  # (i,j) is the coordinate of pre-neuron
#                                 np.clip(int(floor(row + radius)) + 1, 0, dim_pre)):
#                     for j in range(np.clip(int(ceil(col - sqrt(radius ** 2 - (i - row) ** 2))), 0, dim_pre),
#                                     np.clip(int(floor(col + sqrt(radius ** 2 - (i - row) ** 2))) + 1, 0, dim_pre)):
#                         if np.random.rand() < p:
#                             matrix_pre.append(i * dim_pre + j)
#                             matrix_post.append(r * dim_post + c)
#                             distance =(row-i)**2+(col-j)**2
#                             core_weight.append(math.exp(-2.3026*distance/(radius**2)))
#         if len(matrix_pre) == len(matrix_post) and len(core_weight) == len(matrix_post):
#             print('total connections = '+str(len(matrix_pre))+'\naverage = '+str(1. * len(matrix_pre) / dim_pre ** 2)+' per pre-neuron'+
#                   '\naverage = '+str(1. * len(matrix_pre) / dim_post ** 2)+' per post-neuron')
#         else:
#             print('connection index error!!! source dimension does not euqal to target dimension!!!')
#             raise Exception
#     return matrix_pre, matrix_post, np.array(core_weight)
#
#
#
# def figure_spike_map(spikedata, fig_num, groupname=''):  # tested @ 2017-3-9
#     spiking = np.array(spikedata).reshape(int(sqrt(len(spikedata))), int(sqrt(len(spikedata))))
#     fig = figure(fig_num, figsize=(18, 18))
#     im2 = imshow(spiking, interpolation="nearest", vmin=0, vmax=spiking.max(), cmap=cmap.get_cmap('hot_r'))#
#     colorbar(im2)
#     title('spiking of '+groupname)
#     fig.canvas.draw()
#     return im2, fig