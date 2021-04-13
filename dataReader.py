import pandas as pd
import os
import numpy as np
import pickle
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def load_file(filepath):
    print(filepath)
    row_data = pd.read_csv(filepath)
    dataframe=row_data.copy()
    # 选用三轴加速度、三轴角加速度、欧拉角、用分圈标记过的霍尔传感器
    colList = ["accelerationx", "accelerationy", "accelerationz", "angularvelocityx", "angularvelocityy",
               "angularvelocityz", "pitch", "roll", "yaw", "hallsensor"]
    # sns.pairplot(dataframe[colList], diag_kind="kde")
    # plt.show()
    data = dataframe[colList].values
    data = data.astype(np.float64)
    data[:, 0:9] = data[:, 0:9] / 32768.0  # 对前9列数据归一化
    data = np.pad(data, ((0, 1200 - data.shape[0]), (0, 0)), 'constant',
                  constant_values=(0.0, 0.0))  # 将数据填充至1200行
    return data


def load_dataset(dirname, classname, pklPath="./data/pkl"):
    """

    加载数据集，并随机生成训练集与测试集

    :param dirname: 数据集路径
    :param classname: 选择的跳绳评判指标
    :param pklPath: PKL文件路径
    :return: X_train, X_test, y_train, y_test
    """
    X = []
    y0 = []
    y = []

    # 定义gauss噪声的均值和方差
    mu = 0
    sigma = 0.25

    with open(os.path.join(pklPath, "type_2_index.pkl"), 'rb') as f:
        type_2_index = pickle.load(f, encoding='bytes')

    with open(os.path.join(pklPath, "index_2_" + classname + ".pkl"), 'rb') as f:
        index_2_label = pickle.load(f, encoding='bytes')

    fileList = type_2_index[classname]

    print(fileList)

    for file in fileList:
        X.append(load_file(os.path.join(dirname, str(file) + ".csv")))
        y.append([float(index_2_label[file])])
        # y.append(float(index_2_label[file]))
        # y0.append([[float(index_2_label[file])]])

    X = np.array(X)
    y = np.array(y, dtype=np.float64)
    y0 = np.array(y0, dtype=np.float64)

    # plt.scatter(np.arange(y.shape[0]), y.flatten())
    # plt.scatter(np.arange(y0.shape[0]), y0.flatten())
    # plt.show()

    # print(X.shape)
    # print(y.shape)

    # 随机分割测试集与训练集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # 对训练集标签增加高斯噪声
    for i in range(y_train.shape[0]):
        y_train[i][0] += random.gauss(mu, sigma)

    return X_train, X_test, y_train, y_test


def load_dataset2(dirname, classname, pklPath="./data/pkl",augment=False):
    """

    在已固定分割后的测试集与训练集上使用
    功能为加载数据集

    :param dirname: 数据集路径
    :param classname: 选择的跳绳评判指标
    :param pklPath: PKL文件路径
    :return: X_train, X_test, y_train, y_test
    """
    X = []
    y = []

    # 定义gauss噪声的均值和方差
    mu = 0
    sigma = 0.25

    if augment:
        trainList = os.listdir(os.path.join(dirname, classname, "augmentation"))
    else:
        trainList = os.listdir(os.path.join(dirname, classname, "train"))

    testList = os.listdir(os.path.join(dirname, classname, "test"))

    with open(os.path.join(pklPath, "index_2_" + classname + ".pkl"), 'rb') as f:
        index_2_label = pickle.load(f, encoding='bytes')

    if augment:
        for file in trainList:
            X.append(load_file(os.path.join(dirname, classname, "augmentation", file)))
            y.append([float(index_2_label[int(file.split(".")[0].split("_")[0])])])
    else:
        for file in trainList:
            X.append(load_file(os.path.join(dirname, classname, "train", file)))
            y.append([float(index_2_label[int(file.split(".")[0])])])

    X_train = np.array(X)
    y_train = np.array(y, dtype=np.float64)

    X = []
    y = []

    for file in testList:
        X.append(load_file(os.path.join(dirname, classname, "test", file)))
        y.append([float(index_2_label[int(file.split(".")[0])])])

    X_test = np.array(X)
    y_test = np.array(y, dtype=np.float64)

    # plt.scatter(np.arange(y.shape[0]), y.flatten())
    # plt.scatter(np.arange(y0.shape[0]), y0.flatten())
    # plt.show()

    # print(X.shape)
    # print(y.shape)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # 对训练集标签增加高斯噪声
    for i in range(y_train.shape[0]):
        y_train[i][0] += random.gauss(mu, sigma)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset2("./data", "SpeedStability",augment=True)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
