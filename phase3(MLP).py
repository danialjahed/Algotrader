import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# x_train = np.random.random((1000, 3))
# y_train = keras.utils.to_categorical(np.random.randint(3, size=(1000, 1)), num_classes=3)
# x_test = np.random.random((100, 3))
# y_test = keras.utils.to_categorical(np.random.randint(3, size=(100, 1)), num_classes=3)
x_train = pd.read_csv('Data/geneticOutput/output_X_TrainData.csv').as_matrix()
y_train = pd.read_csv('Data/geneticOutput/output_Y_TrainData.csv').as_matrix()
x_test = np.random.random((100, 3))
y_test = keras.utils.to_categorical(np.random.randint(3, size=(100, 1)), num_classes=3)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(3, activation='relu', input_dim=3))
# model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(6, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

model.save('first.h5py')
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
