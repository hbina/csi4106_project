import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "."]).decode("utf8"))
# with open('train.csv', newline='') as csv_file:
#   spam_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
#   for row in spam_reader:
#      a = 0
#    for col in row:
#         if (a != 3):
# print(col, end=" ")
#      a += 1
#  print("")

train = pd.read_csv("train.csv").values
test = pd.read_csv("test.csv").values

# each of the unique item below would have their own input
types_of_pclass = []
types_of_sex = []
types_of_cabin_number = []  # we split cabin ID into 2 seperate parts (cabin letter and cabin number)

for row in train:
    col_iterator = 0
    col_values = []
    for col in row:
        if col_iterator == 0:
            print("the passenger id is", col, "but we dont care about that...")
        elif col_iterator == 1:
            print("did passenger survived?", col, "...this will be the output...")
            col_values.append(col)
        elif col_iterator == 2:
            print("the class of passenger was", col)
            if col == 1:
                col_values.append(1)
                col_values.append(0)
                col_values.append(0)
            elif col == 2:
                col_values.append(0)
                col_values.append(1)
                col_values.append(0)
            elif col == 3:
                col_values.append(0)
                col_values.append(0)
                col_values.append(1)
            else:
                print("some unidentified type? col:", col)
        elif col_iterator == 3:
            print("the name of passenger was", col)
        elif col_iterator == 4:
            print("passenger's gender was", col)
        elif col_iterator == 5:
            print("passenger was", col, "years old")
        elif col_iterator == 6:
            print("passenger have", col, "siblings")
        elif col_iterator == 7:
            print("parch??", col)
        elif col_iterator == 8:
            print("passenger's ticket was", col, "but we dont really care...")
        elif col_iterator == 9:
            print("passenger paid $", col, "for this death trip...")
        elif col_iterator == 10:
            print("passenger's cabin was", col, "...we are going to seperate this into 2 parts; letters and numbers")
        elif col_iterator == 11:
            print("passenger embarked from port", col)
        col_iterator += 1
    print("there are", len(col_values), "elements in col_values")
    print("content:", col_values)
    print("")
