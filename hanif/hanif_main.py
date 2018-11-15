from time import time
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import keras as ks
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def preprocess(input_x):
    irrelevant_features = ['Cabin', 'Embarked', 'Ticket', 'PassengerId']
    for irr_f in irrelevant_features:
        input_x.pop(irr_f)

    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    feature_set = ['Pclass', 'Sex', 'Parch']
    x_train_sf = input_x[feature_set].copy()
    one_hot_encoder.fit(x_train_sf)

    x_train_sf_encoded = one_hot_encoder.transform(x_train_sf)
    x_filtered = input_x[['Fare', 'Age']]

    # Imputer...replaces NaN object with the mean...
    # Obtained from https://stackoverflow.com/a/30319249
    # Create our Imputer to replace missing values with the mean e.g.
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(x_filtered)
    # Impute our data, then train

    x_train_imp = imp.transform(x_filtered)
    x_joined = np.concatenate([x_train_imp, x_train_sf_encoded], axis=1)
    x_normalized = sklearn.preprocessing.normalize(x_joined)
    return x_normalized


X_train = pd.read_csv('../train.csv')
Y_train = X_train.pop('Survived').values
x_train_preprocessed = preprocess(X_train)

X_test_preprocessed = preprocess(pd.read_csv('../test.csv'))
X_test_preprocessed = np.delete(X_test_preprocessed, 12, 1)
seed = 7

k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_scores = []
for train, test in k_fold.split(x_train_preprocessed, Y_train):
    # create model
    model = Sequential()
    model.add(Dense(x_train_preprocessed.shape[1], input_dim=x_train_preprocessed.shape[1], activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    tbCallBack = ks.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                             write_graph=True, write_images=True)
    model.fit(x_train_preprocessed[train], Y_train[train], epochs=100, verbose=1, callbacks=[tbCallBack])
    # evaluate the model
    scores = model.evaluate(x_train_preprocessed[test], Y_train[test], verbose=1)
    print("scores:\n", scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cv_scores.append(scores[1] * 100)
    if scores[1] > 0.6:
        print("summary:")
        model.summary()
        print("weights:")
        print(model.get_weights())
        print("test dataset predictions:")
        print(model.predict(X_test_preprocessed))
print(np.mean(cv_scores), "% +- ", np.std(cv_scores), "%")
