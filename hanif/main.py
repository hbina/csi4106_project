import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import Sequential
from keras.layers import Dense
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

X = pd.read_csv('../dataset/train.csv')

irrelevant_features = ['Cabin', 'Embarked', 'Ticket', 'PassengerId']
for irr_f in irrelevant_features:
    X.pop(irr_f)
Y = X.pop('Survived').values

one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
feature_set = ['Pclass', 'Sex', 'Parch']
X_train_sf = X[feature_set].copy()
one_hot_encoder.fit(X_train_sf)

feature_names = one_hot_encoder.get_feature_names()

X_train_sf_encoded = one_hot_encoder.transform(X_train_sf)
X_filtered = X[['Fare', 'Age']]

# Imputer...replaces NaN object with the mean...
# Obtained from https://stackoverflow.com/a/30319249
# Create our Imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_filtered)
# Impute our data, then train

X_train_imp = imp.transform(X_filtered)
X_joined = np.concatenate([X_train_imp, X_train_sf_encoded], axis=1)
seed = 7
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_scores = []
for train, test in k_fold.split(X_joined, Y):
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=14, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_joined[train], Y[train], epochs=100, verbose=0)
    # evaluate the model
    scores = model.evaluate(X_joined[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cv_scores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
