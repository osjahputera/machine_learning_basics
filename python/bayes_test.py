from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import bayes
import ml_utils

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, shuffle=True)

print('X_train.shape=', X_train.shape)
print('y_train.shape=', y_train.shape)
print('X_test,shape=', X_test.shape)
print('y_test.shape=', y_test.shape)

classifier = bayes.NaiveBayes()
classifier.fit(X_train, y_train)
class_prediction = classifier.predict(X_test)

print('Bayes accuracy=', ml_utils.accuracy(y_test, class_prediction))