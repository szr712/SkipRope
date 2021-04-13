import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimize


def process(d):
    """

    使用欧拉角数据将三轴加速度数据从传感器坐标转换到世界坐标系

    :param d: 所有传感器的原始数据 共9列 分别为三轴加速度、三轴角加速度、欧拉角
    :return: 坐标转换后的数据 格式与输入数据相同
    """
    data = d.copy()
    res = []
    for line in data:
        line[6:] = line[6:] / 18000 * np.pi
        # line[:6] = line[:6] / 32768
        sx = np.sin(line[6])
        sy = np.sin(line[7])
        sz = np.sin(line[8])
        cx = np.cos(line[6])
        cy = np.cos(line[7])
        cz = np.cos(line[8])
        # 旋转矩阵
        R = np.array([[cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
                      [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
                      [-sy, cy * sx, cx * cy]])
        res.append(np.dot(R, np.array(line[0:3]).transpose()).transpose())
    # accWorld = np.concatenate([np.array(d), np.array(res)], axis=1)
    # print(accWorld)
    return np.array(res)


def to_circle(res):
    """

    根据转换成世界坐标系的z轴加速度数据，进行分圈

    :param res: 转换成世界坐标系的加速度计数据
    :return: 一个list，内容是一圈结束时的时间
    """
    res = np.array(res)
    res_mean = np.mean(res[:, 2])
    res += 4096 - res_mean
    # print(np.mean(res[:, 2]))
    close_in_flag = False
    reset_flag = True
    sep = []
    dt = []
    last_dt = 10
    mean_dt = 10
    for i, num in enumerate(res[:, 2]):
        if res[i - 1, 2] > 4000:
            reset_flag = True
        if i > 0:
            if num < res[i - 1, 2]:  # 下降沿
                close_in_flag = True
            elif close_in_flag and res[i - 1, 2] < 3500:  # 下降到3500以下
                if sep:
                    dt = i - 1 - sep[-1]  # 取得时间差
                    if (dt >= max(mean_dt - max(int(mean_dt * 0.6), 5), 10)) and reset_flag:  # 时间差大于一个阈值
                        if len(sep) == 1:
                            mean_dt = dt
                        elif last_dt <= max(mean_dt * 1.5, mean_dt + 15):
                            mean_dt = (mean_dt + last_dt) / 2
                        sep.append(i - 1)
                        reset_flag = False
                        last_dt = dt
                        # print(len(sep), i - 1, last_dt, mean_dt)
                    elif (res[i - 1, 2] < res[sep[-1], 2]) and (dt <= 15):  # 如果还在下降的话，更新刚才入列的点
                        sep[-1] = i - 1
                        if len(sep) > 1:
                            last_dt += dt
                        # print(len(sep), i - 1, last_dt, mean_dt)
                else:
                    if reset_flag:
                        sep.append(i - 1)
                        reset_flag = False
                        # print(len(sep), i - 1, last_dt, mean_dt)
                close_in_flag = False

    # fig1 = plt.figure(1)
    # plt.plot(res[:, 2], lw=0.5)
    # plt.scatter(sep, res[sep, 2], s=1, c='red')
    # plt.show()
    return sep


if __name__ == "__main__":
    srcName = "./data/origin"
    dirName = "./data/circle"

    fileList = os.listdir(srcName)

    for file in fileList:
        df = pd.read_csv(os.path.join(srcName, file))
        sensor = df[df.columns.values.tolist()[1:10]]
        hall = df['hallsensor']
        accWorld = process(sensor.values)  # 坐标转换
        sep = to_circle(accWorld)  # 分圈

        # 改写霍尔传感器数据 一圈内的霍尔数据相同，相邻圈的数据相反，数值为+-1
        hallsensor = -1
        for i in range(1, len(sep)):
            hallsensor = -hallsensor
            for k in range(sep[i - 1], sep[i]):
                hall[k] = hallsensor

        pd.set_option('mode.chained_assignment', None)
        result = np.concatenate([df.values[sep[0]:sep[-1], ], np.array(accWorld)[sep[0]:sep[-1], ]], axis=1)
        dfResult = pd.DataFrame(result)

        # 保存坐标转换后的三轴加速度数据
        dfResult.columns = df.columns.values.tolist() + ["accelerationWorldx", "accelerationWorldy",
                                                         "accelerationWorldz"]
        dfResult.to_csv(os.path.join(dirName, file), index=False)  # 存入csv
        print(dfResult.head(5))

    # result = []
    # accz = []
    #
    # for line in data:
    #     sample = [float(s) for s in line.split(',')]
    #     r, res = process(sample)
    #     result.append(r)
    #     accz.append(res)
    #
    # result = np.array(result)
    # np.savetxt(resultname, result, delimiter=",")
    #
    # to_circle(accz)
