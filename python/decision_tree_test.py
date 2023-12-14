from sklearn import datasets
from sklearn.model_selection import train_test_split

import ml_utils
from decision_tree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print('X_train : ', X_train.shape)
print('X_test : ', X_test.shape)
print('y_train : ', y_train.shape)
print('y_test : ', y_test.shape)

max_depth = 100
min_samples_split = 2
max_features = None

dt = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, max_features=None)
dt.fit(X_train, y_train)

y_predicted = dt.predict(X_test)

print('DT Accuracy : ', ml_utils.accuracy(y_predicted, y_test))

