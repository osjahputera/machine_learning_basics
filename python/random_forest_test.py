from sklearn import datasets
from sklearn.model_selection import train_test_split
import ml_utils
from random_forest import RandomForest
from decision_tree import DecisionTree
import numpy as np
import time

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
n_trees = 5

# Train and test a decision tree
start = time.time()
dt = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, max_features=None)
dt.fit(X_train, y_train)
end = time.time()
print('DT training time= ', end - start)

y_predicted = dt.predict(X_test)
print('DT Accuracy : ', ml_utils.accuracy(y_predicted, y_test))

# Train and test a random forest using n_trees decision trees.
start = time.time()
RF = RandomForest(n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, max_features=None)
RF.fit(X_train, y_train)
end = time.time()
print('RF sequential training time= ', end - start)

# Train random forest using multi-threading
n_threads=5
start = time.time()
RFT = RandomForest(n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split,
                   max_features=None, n_threads=n_threads)
RFT.fit(X_train, y_train)
end = time.time()
print('RF (n_threads=', n_threads, ') training time= ', end - start)

y_predicted = np.array(RF.predict(X_test))
print('RF Accuracy (most common label): ', ml_utils.accuracy(y_predicted, y_test))

y_predicted_probability = np.array(RF.predict_probability(X_test))
# y_predicted_probability is a list of tuple (l,p) where l is the most common label, and p is its probability
# Select y_predicted with probability > 0.5
y_high_prob_idxs = [idx for idx, (label, prob) in enumerate(y_predicted_probability) if prob > 0.5]
print('Predictions with prob > 0.5: ', len(y_high_prob_idxs), " (", len(y_high_prob_idxs)/len(y_test), ")")
print('RF Accuracy (prob > 0.5): ', ml_utils.accuracy(y_predicted[y_high_prob_idxs], y_test[y_high_prob_idxs]))

y_high_prob_idxs = [idx for idx, (label, prob) in enumerate(y_predicted_probability) if prob > 0.6]
print('Predictions with prob > 0.6: ', len(y_high_prob_idxs), " (", len(y_high_prob_idxs)/len(y_test), ")")
print('RF Accuracy (prob > 0.6): ', ml_utils.accuracy(y_predicted[y_high_prob_idxs], y_test[y_high_prob_idxs]))

# You can increase the performance by increasing the probability threshold.
y_high_prob_idxs = [idx for idx, (label, prob) in enumerate(y_predicted_probability) if prob > 0.8]
print('Predictions with prob > 0.8: ', len(y_high_prob_idxs), " (", len(y_high_prob_idxs)/len(y_test), ")")
print('RF Accuracy (prob > 0.8): ', ml_utils.accuracy(y_predicted[y_high_prob_idxs], y_test[y_high_prob_idxs]))

# Test RF trained using multithreading
y_predicted = np.array(RFT.predict(X_test))
print('RFT Accuracy (most common label): ', ml_utils.accuracy(y_predicted, y_test))

y_predicted_probability = np.array(RFT.predict_probability(X_test))
# y_predicted_probability is a list of tuple (l,p) where l is the most common label, and p is its probability
# Select y_predicted with probability > 0.5
y_high_prob_idxs = [idx for idx, (label, prob) in enumerate(y_predicted_probability) if prob > 0.5]
print('Predictions with prob > 0.5: ', len(y_high_prob_idxs), " (", len(y_high_prob_idxs)/len(y_test), ")")
print('RFT Accuracy (prob > 0.5): ', ml_utils.accuracy(y_predicted[y_high_prob_idxs], y_test[y_high_prob_idxs]))

y_high_prob_idxs = [idx for idx, (label, prob) in enumerate(y_predicted_probability) if prob > 0.6]
print('Predictions with prob > 0.6: ', len(y_high_prob_idxs), " (", len(y_high_prob_idxs)/len(y_test), ")")
print('RFT Accuracy (prob > 0.6): ', ml_utils.accuracy(y_predicted[y_high_prob_idxs], y_test[y_high_prob_idxs]))

# You can increase the performance by increasing the probability threshold.
y_high_prob_idxs = [idx for idx, (label, prob) in enumerate(y_predicted_probability) if prob > 0.8]
print('Predictions with prob > 0.8: ', len(y_high_prob_idxs), " (", len(y_high_prob_idxs)/len(y_test), ")")
print('RFT Accuracy (prob > 0.8): ', ml_utils.accuracy(y_predicted[y_high_prob_idxs], y_test[y_high_prob_idxs]))