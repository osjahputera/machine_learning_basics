import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import logistic_regression
import ml_utils

# This is a 2-class problem (suitable for binary logistic regression)
# With 30 features (30-dimensional data)
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, shuffle=True)

print('X_train.shape=', X_train.shape)
print('y_train.shape=', y_train.shape)
print('X_test,shape=', X_test.shape)
print('y_test.shape=', y_test.shape)

lr = 0.0001
iters = 1000

logreg = logistic_regression.LogisticRegression(lr, iters)
logreg.fit(X_train, y_train)
y_predicted = logreg.predict(X_test)

print('accuracy= ', ml_utils.accuracy(y_predicted, y_test))