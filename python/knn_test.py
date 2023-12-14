import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import ml_utils
import knn


# Load Iris data
(iris_data, iris_label) = datasets.load_iris(return_X_y=True)

print('iris_data.shape= ', iris_data.shape)
print('iris_label.shape= ', iris_label.shape)

# Split training from test, use 20% data for testing
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=1234, shuffle=True)

print('train count: ', len(X_train))
print('test count: ', len(X_test))

k_nn = 7
knn_obj = knn.KNN(k_nn)
knn_obj.fit(X_train, y_train)
predicted_labels = knn_obj.predict(X_test)

print('k_nn: ', k_nn)
print('accuracy: ', ml_utils.accuracy(predicted_labels, y_test))

