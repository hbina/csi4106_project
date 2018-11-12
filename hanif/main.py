import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from subprocess import check_output

print(check_output(["ls", "."]).decode("utf8"))

original_train_data = pd.read_csv("../train.csv").values
original_test_data = pd.read_csv("../test.csv").values


model = Sequential()
model.add(Dense(9, input_dim=9, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(36, kernel_initializer='uniform', activation='relu'))
model.add(Dense(18, kernel_initializer='uniform', activation='relu'))
model.add(Dense(9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
model.summary()
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x_as_nparray, train_y_as_nparray,
          epochs=20)
model_prediction = model.predict(test_x_as_nparray, batch_size=400, verbose=0, steps=None)
print("dataset:", test_x_as_nparray)
print("model_prediction:", model_prediction)
