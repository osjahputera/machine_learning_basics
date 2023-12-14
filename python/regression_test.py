import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from regression import LinearRegression, LogisticRegression
import ml_utils

# test LinearRegression
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

linreg = LinearRegression(learn_rate, n_iters)
linreg.fit(X_train, y_train)

y_predicted = linreg.predict(X_test)

print('linreg MSE=', ml_utils.mse(y_test, y_predicted))

fig1 = plt.figure(1, figsize=(4,4))
plt.scatter(X_train[:,0], y_train, color='b', marker='o')
plt.plot(X_test[:,0], y_predicted, color='r')
plt.show()

# Logistic regression test
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, shuffle=True)

print('X_train.shape=', X_train.shape)
print('y_train.shape=', y_train.shape)
print('X_test,shape=', X_test.shape)
print('y_test.shape=', y_test.shape)

lr = 0.0001
iters = 1000

logreg = LogisticRegression(lr, iters)
logreg.fit(X_train, y_train)
y_predicted = logreg.predict(X_test)

print('logreg accuracy= ', ml_utils.accuracy(y_predicted, y_test))

# Try the LogisticRegression from sklearn
# use solver='liblinear' for a 2-class problem
sk_logreg = sklearn.linear_model.LogisticRegression(solver='liblinear', max_iter=1000)
sk_logreg.fit(X_train, y_train)
sk_y_predicted = sk_logreg.predict(X_test)
print(sk_y_predicted.shape)
print('sk accuracy=', ml_utils.accuracy(sk_y_predicted, y_test))