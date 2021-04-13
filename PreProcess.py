import os
import shutil
import pandas as pd
import pickle

srcPath = "./data/circle"
dirPath = "./data/rename"

fileList = os.listdir(srcPath)

# rename 文件 将数据文件的名字设置为index 并存入rename文件夹下
for file in fileList:
    if file[3] == '-':
        shutil.copy(os.path.join(srcPath, file), os.path.join(dirPath, file[0:3] + ".csv"))
    else:
        shutil.copy(os.path.join(srcPath, file), os.path.join(dirPath, file.split('，')[0] + ".csv"))

# 生成字典文件
df = pd.read_csv("./data/label.csv")
dirPath = "./data/pkl"
index_2_class = {}  # 每个index所对应的跳绳类型
class_2_index = {}  # 每个跳绳类型对应的文件list
index_2_SpeedStability = {}  # 每个index对应的SpeedStability值
index_2_PostionStablity = {}  # 每个index对应的PostionStablity值
index_2_RopeSwinging = {}  # 每个index对应的RopeSwinging值
index_2_Coordination = {}  # 每个index对应的Coordination值
type_2_index = {"SpeedStability": [], "PostionStablity": [], "RopeSwinging": [],
                "Coordination": []}  # 每个跳绳评价指标对应的文件list

for index, row in df.iterrows():
    className = row["class"]
    cur = row["id"]

    index_2_class[cur] = className

    if className not in class_2_index.keys():
        class_2_index[className] = []
        class_2_index[className].append(cur)
    else:
        class_2_index[className].append(cur)

    index_2_SpeedStability[cur] = row["SpeedStability"]
    type_2_index["SpeedStability"].append(cur)

    if className == "原地纵跳":
        index_2_PostionStablity[cur] = row["PostionStablity"]
        type_2_index["PostionStablity"].append(cur)
    elif className == "摇绳":
        index_2_RopeSwinging[cur] = row["RopeSwinging"]
        type_2_index["RopeSwinging"].append(cur)
    elif className == "摇跳":
        index_2_PostionStablity[cur] = row["PostionStablity"]
        index_2_RopeSwinging[cur] = row["RopeSwinging"]
        type_2_index["PostionStablity"].append(cur)
        type_2_index["RopeSwinging"].append(cur)
    elif className == "基本单摇跳绳":
        index_2_PostionStablity[cur] = row["PostionStablity"]
        index_2_RopeSwinging[cur] = row["RopeSwinging"]
        index_2_Coordination[cur] = row["Coordination"]
        type_2_index["PostionStablity"].append(cur)
        type_2_index["RopeSwinging"].append(cur)
        type_2_index["Coordination"].append(cur)

with open(os.path.join(dirPath, "index_2_class.pkl"), 'wb') as f:
    pickle.dump(index_2_class, f)
with open(os.path.join(dirPath, "class_2_index.pkl"), 'wb') as f:
    pickle.dump(class_2_index, f)
with open(os.path.join(dirPath, "index_2_SpeedStability.pkl"), 'wb') as f:
    pickle.dump(index_2_SpeedStability, f)
with open(os.path.join(dirPath, "index_2_PostionStablity.pkl"), 'wb') as f:
    pickle.dump(index_2_PostionStablity, f)
with open(os.path.join(dirPath, "index_2_RopeSwinging.pkl"), 'wb') as f:
    pickle.dump(index_2_RopeSwinging, f)
with open(os.path.join(dirPath, "index_2_Coordination.pkl"), 'wb') as f:
    pickle.dump(index_2_Coordination, f)
with open(os.path.join(dirPath, "type_2_index.pkl"), 'wb') as f:
    pickle.dump(type_2_index, f)
