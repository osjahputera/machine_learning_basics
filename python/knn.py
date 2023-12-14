import numpy as np
from collections import Counter
import ml_utils


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X  # training data
        self.y_train = y  # training labels

    def predict(self, X):
        predicted_label = [self._predict_(x) for x in X]
        return np.array(predicted_label)

    def _predict_(self, x):
        # Compute distances from each test data to train data
        distances = [ml_utils.sqr_euclidean_dist(x, x_train) for x_train in self.X_train]

        # Sort the dist in increasing order and select the top k
        k_sorted_idx = np.argsort(distances)[:self.k]

        #print('k_sorted_idx=', k_sorted_idx)

        # Extract the class labels from the nearest k training data
        predicted_labels = [self.y_train[idx] for idx in k_sorted_idx]

        #print('predicted_labels=', predicted_labels)

        # Find the majority among predicted labels
        most_common_label = Counter(predicted_labels).most_common()

        #print('most_common_label=', most_common_label)

        # Return only the label, not the count of the label
        return most_common_label[0][0]
