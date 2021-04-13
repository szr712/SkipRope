# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def process(data):
    """

    分圈，陀曼源码

    :param data: 全部数据
    :return:
    """
    res = []
    for line in data:
        line = np.array(line)
        line[6:] = line[6:] / 18000 * np.pi
        sx = np.sin(line[6])
        sy = np.sin(line[7])
        sz = np.sin(line[8])
        cx = np.cos(line[6])
        cy = np.cos(line[7])
        cz = np.cos(line[8])
        R = np.array([[cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
                      [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
                      [-sy, cy * sx, cx * cy]])
        res.append(np.dot(R, np.array(line[0:3]).transpose()).transpose())
    res = np.array(res)
    res_mean = np.mean(res[:,2])
    res += 4096 - res_mean
    print(np.mean(res[:,2]))
    close_in_flag = False
    reset_flag = True
    sep = []
    dt = []
    last_dt = 10
    mean_dt = 10
    for i, num in enumerate(res[:,2]):
        if res[i-1, 2] > 4000:
            reset_flag = True
        if i > 0:
            if num < res[i-1, 2]: #下降沿
                close_in_flag = True
            elif close_in_flag and res[i-1, 2] < 3500: #下降到3500以下
                if sep:
                    dt = i-1 - sep[-1] #取得时间差
                    if (dt >= max(mean_dt - max(int(mean_dt * 0.6), 5), 10)) and reset_flag:
                        if len(sep) == 1:
                            mean_dt = dt
                        elif last_dt <= max(mean_dt * 1.5, mean_dt + 15):
                            mean_dt = (mean_dt + last_dt) / 2
                        sep.append(i - 1)
                        reset_flag = False
                        last_dt = dt
                        print(len(sep), i - 1, last_dt, mean_dt)
                    elif (res[i-1, 2] < res[sep[-1], 2]) and (dt <= 15):
                        sep[-1] = i - 1
                        if len(sep) > 1:
                            last_dt += dt
                        print(len(sep), i - 1, last_dt, mean_dt)
                else:
                    if reset_flag:
                        sep.append(i - 1)
                        reset_flag = False
                        print(len(sep), i - 1, last_dt, mean_dt)
                close_in_flag = False


    fig1 = plt.figure(1)
    '''plt.subplot(911)
    plt.plot(data[:, 0], lw=0.5)
    plt.subplot(912)
    plt.plot(data[:, 1], lw=0.5)
    plt.subplot(913)
    plt.plot(data[:, 2], lw=0.5)
    plt.subplot(914)
    plt.plot(data[:, 3], lw=0.5)
    plt.subplot(915)
    plt.plot(data[:, 4], lw=0.5)
    plt.subplot(916)
    plt.plot(data[:, 5], lw=0.5)
    plt.subplot(917)
    plt.plot(data[:, 6], lw=0.5)
    plt.subplot(918)
    plt.plot(data[:, 7], lw=0.5)
    plt.subplot(919)
    plt.plot(data[:, 8], lw=0.5)
    fig2 = plt.figure(2)'''
    plt.plot(res[:, 2], lw=0.5)
    plt.scatter(sep, res[sep, 2], s=1, c='red')
    plt.show()


if __name__ == '__main__':
    filename = './data/out2.csv'
    with open(filename, 'r') as f:
        data = f.readlines()
    count = 0
    sample = []
    for line in data:
        sample.append([float(s) for s in line.split(',')])
        count += 1
        if count==7:
            process(sample)
            sample = []
            count =0
        process(sample)
        # if line == '-1234567,-1234567,-1234567,-1234567,-1234567,-1234567,-1234567,-1234567,-1234567\n':
        #     if sample:
        #         if count == 7:
        #             process(sample)
        #     sample = []
        #     count += 1
        #     continue
        # sample.append([float(s) for s in line.split(',')])
    # print(sample)
    #process(sample)
