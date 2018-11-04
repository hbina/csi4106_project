import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "."]).decode("utf8"))

original_train_data = pd.read_csv("train.csv").values
original_test_data = pd.read_csv("test.csv").values

print("parsing train data...")
DEFAULT_AGE = 999
# each of the unique item below would have their own input
train_x_as_list = []
train_y_as_list = []
for row in original_train_data:
    col_iterator = 0
    train_data_x_row_value = []
    train_data_y_row_value = []
    for col in row:
        if col_iterator == 1:
            train_data_y_row_value.append(col)
        elif col_iterator == 2:
            if col == 1:
                train_data_x_row_value.append(1)
                train_data_x_row_value.append(0)
                train_data_x_row_value.append(0)
            elif col == 2:
                train_data_x_row_value.append(0)
                train_data_x_row_value.append(1)
                train_data_x_row_value.append(0)
            elif col == 3:
                train_data_x_row_value.append(0)
                train_data_x_row_value.append(0)
                train_data_x_row_value.append(1)
        elif col_iterator == 4:
            if col == "male":
                train_data_x_row_value.append(1)
                train_data_x_row_value.append(0)
            elif col == "female":
                train_data_x_row_value.append(0)
                train_data_x_row_value.append(1)
        elif col_iterator == 5:
            if (np.isnan(col)):
                train_data_x_row_value.append(DEFAULT_AGE)
            else:
                train_data_x_row_value.append(col)
        elif col_iterator == 6:
            train_data_x_row_value.append(col)
        elif col_iterator == 7:
            train_data_x_row_value.append(col)
        elif col_iterator == 9:
            train_data_x_row_value.append(col)
        col_iterator += 1
    train_x_as_list.append(train_data_x_row_value)
    train_y_as_list.append(train_data_y_row_value)

print("transforming train_x_as_list to np.array")
train_x_as_nparray = np.array(train_x_as_list)
print("transforming train_y_as_list to np.array")
train_y_as_nparray = np.array(train_y_as_list)
print("shape of train_x_as_nparray array is", train_x_as_nparray.shape)
print("shape of train_y_as_nparray array is", train_y_as_nparray.shape)

print("parsing test data...")
test_x_as_list = []
for row in original_test_data:
    col_iterator = 0
    test_data_x_row_value = []
    for col in row:
        if col_iterator == 1:
            if col == 1:
                test_data_x_row_value.append(1)
                test_data_x_row_value.append(0)
                test_data_x_row_value.append(0)
            elif col == 2:
                test_data_x_row_value.append(0)
                test_data_x_row_value.append(1)
                test_data_x_row_value.append(0)
            elif col == 3:
                test_data_x_row_value.append(0)
                test_data_x_row_value.append(0)
                test_data_x_row_value.append(1)
        elif col_iterator == 3:
            if col == "male":
                test_data_x_row_value.append(1)
                test_data_x_row_value.append(0)
            elif col == "female":
                test_data_x_row_value.append(0)
                test_data_x_row_value.append(1)
        elif col_iterator == 4:
            if (np.isnan(col)):
                test_data_x_row_value.append(DEFAULT_AGE)
            else:
                test_data_x_row_value.append(col)
        elif col_iterator == 5:
            test_data_x_row_value.append(col)
        elif col_iterator == 6:
            test_data_x_row_value.append(col)
        elif col_iterator == 8:
            test_data_x_row_value.append(col)
        col_iterator += 1
    test_x_as_list.append(test_data_x_row_value)

print("transforming test_x_as_list to np.array")
test_x_as_nparray = np.array(test_x_as_list)
print("shape of test_x_as_nparray array is", test_x_as_nparray.shape)

print("preparing the network...")
model = Sequential()
K.set_image_dim_ordering('th')
model.add(Dense(12, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.summary()
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x_as_nparray, train_y_as_nparray,
          epochs=20)
print("Kaggle does not provide a annotated test with this...so we would have to generate 1 ourselves...")
# score = model.evaluate(test_x_as_nparray, test_y_as_nparray, batch_size=128)
print("using the model to predict values...")
print("test_x_as_nparray.shape:", test_x_as_nparray.shape)
model_prediction = model.predict(test_x_as_nparray, batch_size=None, verbose=0, steps=None)
print("model_prediction:", model_prediction)
