import numpy as np
import decision_tree as dt

# Create randomized versions of the input data by allowing the same samples
# to be selected multiple times. Since we randomly select N samples out of
# an input dataset containing N samples, and we allow samples to be selected
# multiple times, then we essentially will drop a few samples in the process.
# This is called "bootstrapping"
import ml_utils
import threading


def bootstrap_samples(X, y):
    N = X.shape[0]
    idxs = np.random.choice(a=N, size=N, replace=True)
    return X[idxs], y[idxs]


class RandomForest:
    def __init__(self, n_trees=99, max_depth=100, min_samples_split=2, max_features=None, n_threads=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.the_forest = []
        self.n_threads = np.min(n_threads, n_threads) if not n_threads else n_threads

    def fit(self, X, y):

        # This is a good place to do parallel processing since each tree
        # can be trained independently of each other.
        self.the_forest = []

        if not self.n_threads:
            for _ in range(self.n_trees):
                self._fit_thread(X, y)
        else:
            # Can we apply multi-threading in RF training?
            threads = []
            thread_active = []
            print('n_threads= ', self.n_threads)

            for id in range(self.n_threads):
                thr = threading.Thread(target=self._fit_thread, args=(X, y))
                threads.append(thr)
                thr.start()
                thread_active.append(True)
                print('Starting thread-', id)

            completed_thread = 0

            for i, thr in enumerate(threads):
                thr.join()
                start_new_thread = False
                if thread_active[i]:
                    thread_active[i] = False
                    start_new_thread = True
                    completed_thread += 1
                    print('Thread-', i, ' complete')
                active_thread_count = np.sum(s for s in thread_active if s)
                print('Active thread count = ', active_thread_count)
                print('Completed thread = ', completed_thread)
                if completed_thread + active_thread_count < self.n_trees and active_thread_count < self.n_threads:
                    thr = threading.Thread(target=self._fit_thread, args=(X, y))
                    threads.append(thr)
                    thr.start()
                    thread_active.append(True)
                    print('Starting a new thread, active_thread_count = ', active_thread_count+1)

            print('All threads complete, self.the_forest has ', len(self.the_forest), ' trained trees')

    def _fit_thread(self, X, y):
        X_boot, y_boot = bootstrap_samples(X, y)
        tree = dt.DecisionTree(self.max_depth, min_samples_split=self.min_samples_split,
                               max_features=self.max_features)
        tree.fit(X_boot, y_boot)
        self.the_forest.append(tree)

    def predict(self, X):
        tree_pred = np.array([tree.predict(X) for tree in self.the_forest])
        # tree_pred is a list of size n_trees, each element is a list of n_samples, so
        # tree_pred[i][j] is the prediction from tree-i for sample-j
        # We want to reshape this into n_samples x n_trees, so tree_pred[j] contains
        # the predictions for sample-j from all trees.
        sample_prediction = [ml_utils.most_common_label(tree_opinions) for tree_opinions in tree_pred.T]
        return sample_prediction

    def predict_probability(self, X):
        tree_pred = np.array([tree.predict(X) for tree in self.the_forest])
        # tree_pred is a list of size n_trees, each element is a list of n_samples, so
        # tree_pred[i][j] is the prediction from tree-i for sample-j
        # We want to reshape this into n_samples x n_trees, so tree_pred[j] contains
        # the predictions for sample-j from all trees.
        sample_prediction_probability = [ml_utils.most_common_label_probability(tree_opinions) for tree_opinions in tree_pred.T]
        return sample_prediction_probability


