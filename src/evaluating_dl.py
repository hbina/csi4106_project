# MLP for Pima Indians Dataset with 10-fold cross validation
import numpy
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# define 10-fold cross validation test harness
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_scores = []
for train, test in k_fold.split(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cv_scores.append(scores[1] * 100)
print(numpy.mean(cv_scores), "(+/- ", numpy.std(cv_scores), ")")
