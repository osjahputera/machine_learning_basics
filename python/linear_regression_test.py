import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import linear_regression
import ml_utils

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, shuffle=True)

print('X_train.shape=', X_train.shape)
print('y_train.shape=', y_train.shape)
print('X_test,shape=', X_test.shape)
print('y_test.shape=', y_test.shape)

plt.scatter(X[:,0], y, color='b', marker='o')
#plt.show()

learn_rate = 0.01
n_iters = 1000

linreg = linear_regression.LinearRegression()
linreg.fit(X_train, y_train, learn_rate, n_iters)

y_predicted = linreg.predict(X_test)

print('MSE=', ml_utils.mse(y_test, y_predicted))

plt.scatter(X_train[:,0], y_train, color='b', marker='o')
plt.plot(X_test[:,0], y_predicted, color='r')
plt.show()