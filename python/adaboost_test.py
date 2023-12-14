from sklearn import datasets
from sklearn.model_selection import train_test_split
import ml_utils
from adaboost import AdaBoost

data = datasets.load_breast_cancer()
X, y = data.data, data.target

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# Adaboost classification with 5 weak classifiers
clf = AdaBoost(n_classifiers=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = ml_utils.accuracy(y_test, y_pred)
print("Accuracy:", acc)