import random

import pandas as pd
import numpy as np
import os
import shutil

dataSet = "./data"
className = "SpeedStability"
augDataPath = "augmentation"


def to_circle(circleList):
    result = []
    hallsensor = 1
    for circle in circleList:
        c = circle.copy()
        c[:, 11] = hallsensor
        hallsensor = -hallsensor
        result.append(c)
    return result


def augment_data(data):
    augData = []
    hallsensor = -1
    circleList = []
    pre = 0
    for i in range(0, data.shape[0]):
        if data[i][11] == hallsensor:
            hallsensor = -hallsensor
            # 切割进入list
            circleList.append(data[pre:i])
            pre = i
    circleList.append(data[pre:])
    # li = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # for i in range(0, 15):
    #     random.shuffle(li)
    #     l = li.copy()
    #     augData.append(l)
    # print(augData)
    for i in range(0, 14):
        random.shuffle(circleList)
        result = to_circle(circleList.copy())  # 重新分圈，以防出现数圈hallsensor为1时被并为同圈的情况
        augData.append(result.copy())
    return augData


def load_file(fileName):
    dataframe = pd.read_csv(os.path.join(dataSet, className, "train", fileName))
    print(fileName)
    # 选用三轴加速度、三轴角加速度、欧拉角、用分圈标记过的霍尔传感器
    # colList = ["accelerationx", "accelerationy", "accelerationz", "angularvelocityx", "angularvelocityy",
    #            "angularvelocityz", "pitch", "roll", "yaw", "hallsensor"]
    data = dataframe.values
    # data = data.astype(np.float64)
    augData = augment_data(data)
    for i, aug in enumerate(augData):
        aug = np.vstack(aug)
        dfAug = pd.DataFrame(aug)
        dfAug.columns = dataframe.columns.values.tolist()
        dfAug.to_csv(os.path.join(dataSet, className, augDataPath, fileName.split(".")[0] + "_" + str(i) + ".csv"),
                     index=False)  # 存入csv
    # 最后把原始数据复制过去
    shutil.copy(os.path.join(dataSet, className, "train", fileName),
                os.path.join(dataSet, className, augDataPath, fileName.split(".")[0] + "_15.csv"))


if __name__ == "__main__":
    for file in os.listdir(os.path.join(dataSet, className, "train")):
        load_file(file)
