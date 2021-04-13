from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.metrics import MeanSquaredError

model = Sequential()
model.add(LSTM(100, input_shape=(1200, 10)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='sgd', metrics=[MeanSquaredError()])

model.summary()