# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot

# S2的发放率
def S2_firing_rate():
    S2 = 0.7*np.random.random([55,55])
    # 设置中心值,12行，14列
    row = 27
    col = 34
    S2[row,col] = 11.5
    base = 11.5
    factor = 1.
    S2[row,col+1] = base-factor*np.random.rand(1)[0]
    S2[row,col-1] = base-factor*np.random.rand(1)[0]
    S2[row-1,col] = base-factor*np.random.rand(1)[0]
    S2[row+1,col] = base-factor*np.random.rand(1)[0]
    S2[row-1,col-1] = base-factor*np.random.rand(1)[0]
    S2[row-1,col+1] = base-factor*np.random.rand(1)[0]
    S2[row+1,col-1] = base-factor*np.random.rand(1)[0]
    S2[row+1,col+1] = base-factor*np.random.rand(1)[0]

    base = 11
    factor = 1.
    line = 2
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 10.5
    factor = 1
    line = 3
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 9.5
    factor = 1
    line = 4
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 9
    factor = 1
    line = 5
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 8
    factor = 1.5
    line = 6
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 7
    factor = 1.5
    line = 7
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 6
    factor = 3
    line = 8
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 3
    factor = 2
    line = 9
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 2
    factor = 2
    line = 10
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]


    # 另一团
    row = 32
    col = 24
    S2[row,col] = 10.5
    base = 10.5
    factor = 1
    S2[row,col+1] = base-factor*np.random.rand(1)[0]
    S2[row,col-1] = base-factor*np.random.rand(1)[0]
    S2[row-1,col] = base-factor*np.random.rand(1)[0]
    S2[row+1,col] = base-factor*np.random.rand(1)[0]
    S2[row-1,col-1] = base-factor*np.random.rand(1)[0]
    S2[row-1,col+1] = base-factor*np.random.rand(1)[0]
    S2[row+1,col-1] = base-factor*np.random.rand(1)[0]
    S2[row+1,col+1] = base-factor*np.random.rand(1)[0]

    base = 9.5
    factor = 1.3
    line = 2
    for i in range(-line,line+1):
        S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = base - factor * np.random.rand(1)[0]

    base = 8.5
    factor = 1.5
    line = 3
    for i in range(-line,line+1):
        S2[row-line,col+i] = 8.5-2*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = 9.5 - 1.5 * np.random.rand(1)[0]

    base = 7
    factor = 3
    line = 4
    for i in range(-line,line+1):
        S2[row-line,col+i] = 8.5-3*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = 9 - 1.5 * np.random.rand(1)[0]

    base = 4
    factor = 4
    line = 5
    for i in range(-line,line+1):
        S2[row-line,col+i] = 8.5-3*np.random.rand(1)[0]
        S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    for i in range(-line+1,line):
        S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
        S2[row +i, col + line] = 9 - 1.5 * np.random.rand(1)[0]
    #
    # base = 3
    # factor = 3
    # line = 6
    # for i in range(-line,line+1):
    #     S2[row-line,col+i] = base-factor*np.random.rand(1)[0]
    #     S2[row +line, col + i] = base-factor*np.random.rand(1)[0]
    # for i in range(-line+1,line):
    #     S2[row +i, col - line] = base - factor * np.random.rand(1)[0]
    #     S2[row +i, col + line] = base - factor * np.random.rand(1)[0]
    # 显示画布
    fig1 = pyplot.figure(1, figsize=(5, 5))
    # 显示图形，定义不同类型的colormap
    im1 = pyplot.imshow(S2, interpolation="nearest", vmin=0, vmax=12, cmap='jet')
    pyplot.colorbar(im1)  # 显示colorbar
    pyplot.title('S2 rate')
    fig1.savefig('S2.jpg')
    # pyplot.close(fig)
    pyplot.show()

def S3_firing_rate():
    S3 = 0.9*np.random.random([14,14])
    # 设置中心值,12行，14列
    row = 7
    col = 9
    S3[row,col] = 11.5
    base = 11.5
    factor = 2
    S3[row,col+1] = base-factor*np.random.rand(1)[0]
    S3[row,col-1] = base-factor*np.random.rand(1)[0]
    S3[row-1,col] = base-factor*np.random.rand(1)[0]
    S3[row+1,col] = base-factor*np.random.rand(1)[0]
    S3[row-1,col-1] = base-factor*np.random.rand(1)[0]
    S3[row-1,col+1] = base-factor*np.random.rand(1)[0]
    S3[row+1,col-1] = base-factor*np.random.rand(1)[0]
    S3[row+1,col+1] = base-factor*np.random.rand(1)[0]

    base = 10
    factor = 3.5
    for i in range(-2,3): # [-2,2]
        S3[row-2,col+i] = base-factor*np.random.rand(1)[0]
        S3[row + 2, col + i] = base - factor*np.random.rand(1)[0]
    S3[row-1,col-2] = base-factor*np.random.rand(1)[0]
    S3[row-1,col+2] = base-factor*np.random.rand(1)[0]

    S3[row,col-2] = base-factor*np.random.rand(1)[0]
    S3[row,col+2] = base-factor*np.random.rand(1)[0]

    S3[row+1,col-2] = base-factor*np.random.rand(1)[0]
    S3[row+1,col+2] = base-factor*np.random.rand(1)[0]

    # base = 4.
    # factor = 4.
    # for i in range(-3,4): # [-3,3]
    #     S3[row-3,col+i] = base-factor*np.random.rand(1)[0]
    #     S3[row + 3, col + i] = base-factor*np.random.rand(1)[0]
    # S3[row-2,col-3] = base-factor*np.random.rand(1)[0]
    # S3[row-2,col+3] = base-factor*np.random.rand(1)[0]
    # S3[row-1,col-3] = base-factor*np.random.rand(1)[0]
    # S3[row-1,col+3] = base-factor*np.random.rand(1)[0]
    # S3[row,col-3] = base-factor*np.random.rand(1)[0]
    # S3[row,col+3] = base-factor*np.random.rand(1)[0]
    # S3[row+1,col-3] = base-factor*np.random.rand(1)[0]
    # S3[row+1,col+3] = base-factor*np.random.rand(1)[0]
    # S3[row+2,col-3] = base-factor*np.random.rand(1)[0]
    # S3[row+2,col+3] = base-factor*np.random.rand(1)[0]

    # base = 7
    # factor = 7.
    # line = 4
    # for i in range(-line,line+1):
    #     S3[row-line,col+i] = base-factor*np.random.rand(1)[0]
    #     S3[row +line, col + i] = base-factor*np.random.rand(1)[0]
    # for i in range(-line+1,line):
    #     S3[row +i, col - line] = base - factor * np.random.rand(1)[0]
    #     S3[row +i, col + line] = base - factor * np.random.rand(1)[0]

    # base = 6
    # factor = 4.
    # line = 5
    # for i in range(-line,line+1):
    #     S3[row-line,col+i] = base-factor*np.random.rand(1)[0]
    #     S3[row +line, col + i] = base-factor*np.random.rand(1)[0]
    # for i in range(-line+1,line):
    #     S3[row +i, col - line] = base - factor * np.random.rand(1)[0]
    #     S3[row +i, col + line] = base - factor * np.random.rand(1)[0]
    #
    # base = 2
    # factor = 2.
    # line = 6
    # for i in range(-line,line+1):
    #     S3[row-line,col+i] = base-factor*np.random.rand(1)[0]
    #     S3[row +line, col + i] = base-factor*np.random.rand(1)[0]
    # for i in range(-line+1,line):
    #     S3[row +i, col - line] = base - factor * np.random.rand(1)[0]
    #     S3[row +i, col + line] = base - factor * np.random.rand(1)[0]

    # base = 6
    # factor = 3
    # for i in range(-4,5): # [-4,4]
    #     S3[row-4,col+i] = base-factor*np.random.rand(1)[0]
    #     S3[row + 4, col + i] = base-factor*np.random.rand(1)[0]
    # S3[row-3,col-4] = base-factor*np.random.rand(1)[0]
    # S3[row-3,col+4] = base-factor*np.random.rand(1)[0]
    # S3[row-2,col-4] = base-factor*np.random.rand(1)[0]
    # S3[row-2,col+4] = base-factor*np.random.rand(1)[0]
    # S3[row-1,col-4] = base-factor*np.random.rand(1)[0]
    # S3[row-1,col+4] = base-factor*np.random.rand(1)[0]
    # S3[row,col-4] = base-factor*np.random.rand(1)[0]
    # S3[row,col+4] = base-factor*np.random.rand(1)[0]
    # S3[row+1,col-4] = base-factor*np.random.rand(1)[0]
    # S3[row+1,col+4] = base-factor*np.random.rand(1)[0]
    # S3[row+2,col-4] = base-factor*np.random.rand(1)[0]
    # S3[row+2,col+4] = base-factor*np.random.rand(1)[0]
    # S3[row+3,col-4] = base-factor*np.random.rand(1)[0]
    # S3[row+3,col+4] = base-factor*np.random.rand(1)[0]

    # base = 3
    # factor = 3
    # for i in range(-5,6): # [-5,5]
    #     S3[row-5,col+i] = base-factor*np.random.rand(1)[0]
    #     S3[row + 5, col + i] = base-factor*np.random.rand(1)[0]
    # S3[row - 4, col - 5] = base - factor * np.random.rand(1)[0]
    # S3[row - 4, col + 5] = base - factor * np.random.rand(1)[0]
    # S3[row-3,col-5] = base-factor*np.random.rand(1)[0]
    # S3[row-3,col+5] = base-factor*np.random.rand(1)[0]
    # S3[row-2,col-5] = base-factor*np.random.rand(1)[0]
    # S[row-2,col+5] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col-5] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col+5] = base-factor*np.random.rand(1)[0]
    # S2[row,col-5] = base-factor*np.random.rand(1)[0]
    # S2[row,col+5] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col-5] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col+5] = base-factor*np.random.rand(1)[0]
    # S2[row+2,col-5] = base-factor*np.random.rand(1)[0]
    # S2[row+2,col+5] = base-factor*np.random.rand(1)[0]
    # S2[row+3,col-5] = base-factor*np.random.rand(1)[0]
    # S2[row+3,col+5] = base-factor*np.random.rand(1)[0]
    # S2[row+4,col-5] = base-factor*np.random.rand(1)[0]
    # S2[row+4,col+5] = base-factor*np.random.rand(1)[0]
    #
    # # 另一团
    # row = 32
    # col = 28
    # S2[row,col] = 10.5
    # base = 10.5
    # factor = 2
    # S2[row,col+1] = base-factor*np.random.rand(1)[0]
    # S2[row,col-1] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col-1] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col+1] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col-1] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col+1] = base-factor*np.random.rand(1)[0]
    #
    # base = 8.6
    # factor = 2
    # for i in range(-2,3): # [-2,2]
    #     S2[row-2,col+i] = base-factor*np.random.rand(1)[0]
    #     S2[row + 2, col + i] = base - factor*np.random.rand(1)[0]
    # S2[row-1,col-2] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col+2] = base-factor*np.random.rand(1)[0]
    #
    # S2[row,col-2] = base-factor*np.random.rand(1)[0]
    # S2[row,col+2] = base-factor*np.random.rand(1)[0]
    #
    # S2[row+1,col-2] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col+2] = base-factor*np.random.rand(1)[0]
    #
    # base = 5.5
    # factor = 5
    # for i in range(-3,4): # [-3,3]
    #     S2[row-3,col+i] = base-factor*np.random.rand(1)[0]
    #     S2[row + 3, col + i] = base-factor*np.random.rand(1)[0]
    # S2[row-2,col-3] = base-factor*np.random.rand(1)[0]
    # S2[row-2,col+3] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col-3] = base-factor*np.random.rand(1)[0]
    # S2[row-1,col+3] = base-factor*np.random.rand(1)[0]
    # S2[row,col-3] = base-factor*np.random.rand(1)[0]
    # S2[row,col+3] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col-3] = base-factor*np.random.rand(1)[0]
    # S2[row+1,col+3] = base-factor*np.random.rand(1)[0]
    # S2[row+2,col-3] = base-factor*np.random.rand(1)[0]
    # S2[row+2,col+3] = base-factor*np.random.rand(1)[0]

    # 显示画布
    fig1 = pyplot.figure(1, figsize=(5, 5))
    # 显示图形，定义不同类型的colormap
    im1 = pyplot.imshow(S3, interpolation="nearest", vmin=0, vmax=12, cmap='jet')
    pyplot.colorbar(im1)  # 显示colorbar
    pyplot.title('S3 rate')
    fig1.savefig('S33.jpg')
    # pyplot.close(fig)
    pyplot.show()

def C3_firing_rate():
    S3 = np.random.random([14,1])
    # 设置中心值,12行，14列
    S3[13,0] = 11.
    S3[11,0] = 9.5
    S3[5,0] = 10
    S3[2,0] = 9.8
    S3[10,0] = 9
    S3[3,0] = 7.9
    # 显示画布
    fig1 = pyplot.figure(1, figsize=(5, 5))
    # 显示图形，定义不同类型的colormap
    im1 = pyplot.imshow(S3, interpolation="nearest", vmin=0, vmax=12, cmap='jet')
    pyplot.colorbar(im1)  # 显示colorbar
    pyplot.title('C3 rate')
    fig1.savefig('C3.jpg')
    # pyplot.close(fig)
    pyplot.show()

def S1_weights():
    S3 = np.zeros([7,7])
    # 设置中心值,12行，14列
    for i in range(7):
        S3[i,6-i] = 10
    for i in range(6):
        S3[i,5-i] = 2.
        S3[6-i,i+1] = 2.
    for i in range(5):
        S3[i,4-i] = 1
        S3[6-i,i+2] = 1
    # S3[:,3] = 10
    # S3[:,2] = 1.5
    # S3[:,4] = 1.5
    # S3[:,1] = 0.8
    # S3[:,5] = 0.8
    # S3[3,:] = 10
    # S3[2,:] = 1.5
    # S3[4,:] = 1.5
    # S3[1,:] = 0.8
    # S3[5,:] = 0.8
    # for i in range(7):
    #     S3[i,i] = 10
    # for i in range(6):
    #     S3[i,i+1] = 2.
    #     S3[6-i,5-i] = 2.
    # for i in range(5):
    #     S3[i,i+2] = 1
    #     S3[6-i,4-i] = 1
    # 显示画布
    fig1 = pyplot.figure(1, figsize=(5, 5))
    # 显示图形，定义不同类型的colormap
    im1 = pyplot.imshow(S3, interpolation="nearest", vmin=0, vmax=10, cmap='gray')
    pyplot.colorbar(im1)  # 显示colorbar
    pyplot.title('C3 rate')
    fig1.savefig('S1_weights.jpg')
    # pyplot.close(fig)
    pyplot.show()

S1_weights()
# S3的发放率