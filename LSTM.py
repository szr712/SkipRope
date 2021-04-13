from datetime import datetime

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os

from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from dataReader import load_dataset, load_dataset2

modelName = "扩容_LSTM单层_优化参数_"

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
epochs, batch_size = 200, 32
dataSet = "./data"
className = "SpeedStability"
logDir = "./logs"
curTime = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
modelPath = "./model"
augment = True


def create_model():
    model = Sequential()
    model.add(LSTM(352, input_shape=(1200, 10)))
    model.add(Dropout(0.5))
    # model.add(LSTM(100))
    # model.add(Dropout(0.5))
    model.add(Dense(320, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.001), metrics=['mse', 'mae'])
    model.summary()
    return model


def create_model2():
    model = Sequential()
    model.add(LSTM(160, input_shape=(1200, 10), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.001), metrics=['mse', 'mae'])
    model.summary()
    return model


def get_callbacks():
    return [
        callbacks.EarlyStopping(monitor='mse', patience=40, restore_best_weights=True),
        callbacks.TensorBoard(log_dir=os.path.join(logDir, className, modelName + curTime)),
    ]


def train_model(model, trainX, trainy, testX, testy):
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(testX, testy),
                        callbacks=get_callbacks())
    result = model.evaluate(testX, testy, batch_size=batch_size)
    return history, result


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()


def plot_predict(model, testX, testy):
    test_predictions = model.predict(testX)
    # print(test_predictions)
    # print(testy)
    plt.figure()
    plt.scatter(testy, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    _ = plt.plot([-100, 100], [-100, 100])

    plt.figure()
    error = test_predictions - testy
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset2(dataSet, className, augment=augment)

    model = create_model()
    # model = create_model2()

    history, result = train_model(model, X_train, y_train, X_test, y_test)

    saveName = modelName + str(round(result[0], 3)) + "_" + curTime + ".h5"
    model.save(os.path.join(modelPath, className, saveName))

    # plot_history(history)
    # plot_predict(model, X_test, y_test)
    # plt.show()
