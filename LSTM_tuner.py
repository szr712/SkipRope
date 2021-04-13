from datetime import datetime

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os

from tensorflow.python.keras.metrics import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
import kerastuner as kt

from dataReader import load_dataset, load_dataset2

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
epochs, batch_size = 70, 32
dataSet = "./data"
className = "SpeedStability"
logDir = "./logs"
curTime = datetime.now().strftime("%Y%m%d_%H_%M_%S")
modelPath = "./model"


def create_model(hp):
    model = Sequential()
    hp_units = hp.Int('units1', min_value=32, max_value=512, step=32)
    model.add(LSTM(units=hp_units, input_shape=(1200, 10)))
    model.add(Dropout(0.5))
    # model.add(LSTM(100))
    # model.add(Dropout(0.5))
    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
    model.add(Dense(hp_units2, activation='relu'))
    model.add(Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='mse', optimizer=RMSprop(learning_rate=hp_learning_rate), metrics=['mse', 'mae'])
    model.summary()
    return model

def create_model2(hp):
    model = Sequential()
    hp_units = hp.Int('units1', min_value=32, max_value=512, step=32)
    model.add(LSTM(units=hp_units, input_shape=(1200, 10),return_sequences=True))
    model.add(Dropout(0.5))
    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
    model.add(LSTM(units=hp_units2))
    model.add(Dropout(0.5))
    hp_units3 = hp.Int('units3', min_value=32, max_value=512, step=32)
    model.add(Dense(hp_units3, activation='relu'))
    model.add(Dense(1))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='mse', optimizer=RMSprop(learning_rate=hp_learning_rate), metrics=['mse', 'mae'])
    model.summary()
    return model


def get_callbacks():
    return [
        callbacks.EarlyStopping(monitor='mse', patience=10, restore_best_weights=True),
        callbacks.TensorBoard(log_dir=os.path.join(logDir, className, curTime)),
    ]


def train_model(model, trainX, trainy, testX, testy):
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(testX, testy),
                        callbacks=get_callbacks())
    model.evaluate(testX, testy, batch_size=batch_size)
    return history


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
    X_train, X_test, y_train, y_test = load_dataset2(dataSet, className,augment=True)

    tuner = kt.Hyperband(create_model2,
                         objective='val_loss',  # 优化的目标
                         max_epochs=200,  # 最大迭代次数
                         factor=3,
                         directory='./logs/kerasTuner_aug2',
                         project_name='intro_to_kt')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. 
    unites1:{best_hps.get('units1')} 
    unites2:{best_hps.get('units2')} 
    learning_rate:{best_hps.get('learning_rate')}
    """)

    # model = tuner.hypermodel.build(best_hps)
    #
    # # model = create_model()
    # #
    # history = train_model(model, X_train, y_train, X_test, y_test)
    #
    # model.save(os.path.join(modelPath, className, curTime + ".h5"))

    # plot_history(history)
    # plot_predict(model, X_test, y_test)
    # plt.show()
