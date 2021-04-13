import os

from tensorflow.python.keras.models import load_model

from LSTM import plot_predict
from dataReader import load_dataset2
import matplotlib.pyplot as plt

modelName = "扩容_LSTM双层_1.755__20210413_15_34_02.h5"
modelPath = "./model"
className = "SpeedStability"

model = load_model(os.path.join(modelPath, className, modelName))
model.summary()

X_train, X_test, y_train, y_test = load_dataset2("./data", className)

model.evaluate(X_test, y_test)

plot_predict(model, X_test, y_test)
plt.show()
