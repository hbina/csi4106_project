import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

print("opening dataset file....")
X = pd.read_csv('../dataset/train.csv')
print("obtaining the annotated result...")
Y = X.pop('Survived').values
print("splitting dataset into training and test")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

print("shape of X", X_train.shape)
print("shape of Y", y_test.shape)

print("performing one hot encoding on the input X")
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
feature_set = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

print("filtering out irrelevant features...")
X_train_sf = X_train[feature_set].copy()
X_test_sf = X_test[feature_set].copy()

# Imputer...replaces NaN object with the mean...
# Obtained from https://stackoverflow.com/a/30319249
# Create our imputer to replace missing values with the mean e.g.
print("using Imputer to convert unknown values to mean")
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train_sf)
# Impute our data, then train
X_train_imp = imp.transform(X_train_sf)
print("imputed training set")
print(X_train_imp)

one_hot_encoder.fit(X_train_imp)

feature_names = one_hot_encoder.get_feature_names()
print("the following are the features that have been encoded")
print(feature_names)

X_train_sf_encoded = one_hot_encoder.transform(X_train_imp)
X_test_sf_encoded = one_hot_encoder.transform(X_test_sf)

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
model.fit(X_train_sf_encoded, y_train,
          epochs=20)
