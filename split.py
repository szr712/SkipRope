import os
import pickle
import shutil

from sklearn.model_selection import train_test_split


def split(classname):
    """

    为各类跳绳评判指标切割出固定的测试集与训练集
    并存入指定位置

    :param classname: 选择的跳绳评判指标
    :return:
    """
    srcPath = "./data/rename"
    pklPath = "./data/pkl"

    with open(os.path.join(pklPath, "type_2_index.pkl"), 'rb') as f:
        type_2_index = pickle.load(f, encoding='bytes')

    with open(os.path.join(pklPath, "index_2_" + classname + ".pkl"), 'rb') as f:
        index_2_label = pickle.load(f, encoding='bytes')

    fileList = type_2_index[classname]

    trainList, testList = train_test_split(fileList, test_size=0.33)

    # print(trainList)
    # print(testList)
    for file in trainList:
        shutil.copy(os.path.join(srcPath, str(file) + ".csv"),
                    os.path.join("./data", classname, "train", str(file) + ".csv"))
    for file in testList:
        shutil.copy(os.path.join(srcPath, str(file) + ".csv"),
                    os.path.join("./data", classname, "test", str(file) + ".csv"))


if __name__ == "__main__":
    split("SpeedStability")
