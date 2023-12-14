import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import perceptron
import numpy as np
import ml_utils

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, shuffle=True)

print('X_train.shape=', X_train.shape)
print('y_train.shape=', y_train.shape)
print('X_test,shape=', X_test.shape)
print('y_test.shape=', y_test.shape)

lr = 0.01
iters = 1000

P = perceptron.Perceptron(lr, iters)
P.fit(X_train, y_train)
y_predicted = P.predict(X_test)
y_test_scaled = np.array([1 if i > 0 else 0 for i in y_test])
print("Perceptron accuracy=", ml_utils.accuracy(y_test_scaled, y_predicted))
#print("Perceptron accuracy=", ml_utils.accuracy(y_test, y_predicted))

fig = plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, marker='o')

# We now want to draw the decision boundary defined by the learned W of the Perceptron.
# Suppose we have x0_A and x0_B as the min and max value of X_train[:,0]
# We need to find the corresponding x1_A and x1_B for these points based on the model y = W*X - b
# To be on the line, the following condition must be satisfied: W*X + b = 0
# W[0] * x0_A + W[1] * x1_A + b = 0  --> x1_A = (-W[0] * x0_A - b) / W[1]
# W[0] * x0_B + W[1] * x1_B + b = 0  --> x1_B = (-W[0] * x0_B - b) / W[1]

x0_A = np.amin(X_train[:,0])
x0_B = np.amax(X_train[:,0])

x1_A = -(P.bias + (P.W[0]*x0_A))/P.W[1]
x1_B = -(P.bias + (P.W[0]*x0_B))/P.W[1]

plt.plot([x0_A, x0_B], [x1_A, x1_B], color='r')

# plot test data
test_color = ['green' if i == 0 else 'black' for i in y_predicted]
plt.scatter(X_test[:,0], X_test[:,1], c=test_color, marker='*')

plt.show()


