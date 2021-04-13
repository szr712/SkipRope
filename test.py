import os

from tensorflow.python.keras.models import load_model

from dataReader import load_dataset2

modelName="无扩容_LSTM单层_1.741520881652832__20210413_11_12_31.h5"
modelPath="./model"
className="SpeedStability"

model=load_model(os.path.join(modelPath,className,modelName))
model.summary()

X_train, X_test, y_train, y_test = load_dataset2("./data", className)

model.evaluate(X_test,y_test)