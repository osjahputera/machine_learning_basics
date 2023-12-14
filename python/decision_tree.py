import numpy as np
import ml_utils

def split_node(X_column, threshold):
    # Create the split using threshold.
    #left_child_idxs = [idx for idx, v in enumerate(X_column) if v <= threshold]
    #right_child_idxs = [idx for idx, v in enumerate(X_column) if v > threshold]
    # A better way is to use numpy.argwhere()
    left_child_idxs = np.argwhere(X_column <= threshold).flatten()
    right_child_idxs = np.argwhere(X_column > threshold).flatten()
    return left_child_idxs, right_child_idxs

def entropy(y):
    class_probability = np.bincount(y) / float(len(y))
    # Note that log2(p) does not exist for p <= 0, so explicitly check for p > 0
    return -np.sum([p*np.log2(p) for p in class_probability if p > 0])

def information_gain(X_column, y, threshold):
    class_probability = np.bincount(y) / float(len(y))

    # Calculate parent entropy
    E_parent = entropy(y)

    # Get the child split
    left_child_idxs, right_child_idxs = split_node(X_column, threshold)

    if len(left_child_idxs) == 0 or len(right_child_idxs) == 0:
        return 0

    # Calculate child entropy
    E_left = entropy(y[left_child_idxs])
    E_right = entropy(y[right_child_idxs])

    # Weighted sum of child entropy
    n = len(y)
    E_child = (len(left_child_idxs)/n)*E_left + (len(right_child_idxs)/n)*E_right

    info_gain = E_parent -E_child

    return info_gain


# Decision Tree Node class.
class DT_Node:
    def __init__(self, feature=None, feat_threshold=None, left_child=None, right_child=None, *, leaf_value=None):
        # The * before the parameter leaf_value is added to require user to explicitly specify leaf_value by
        # parameter name. This will secure the instantiation of a leaf node by providing leaf_value != None
        self.feature = feature
        self.feat_threshold = feat_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_value = leaf_value

    def is_leaf_node(self):
        return self.leaf_value is not None


class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # If max_features is set, we randomly pick this many features to evaluate
        # as determinant to find the best feature and threshold to split a node.
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        # Grow the tree
        # Start from the root
        n_samples, n_features = X.shape

        self.max_features = np.min(self.max_features, n_features) if self.max_features is not None else n_features
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        classes = np.unique(y)

        # Check for stopping criteria:
        if (depth >= self.max_depth or num_samples < self.min_samples_split
            or len(classes) == 1):
            # We are at a leaf node!
            leaf_value = ml_utils.most_common_label(y)
            return DT_Node(leaf_value=leaf_value)

        # Randomly select self.max_features feature idxs from [0, ..., num_features-1],
        # so we will evaluate only max_features of features at a time. replace=False so
        # once an idx is picked it cannot be picked again (no duplicate idx in the result)
        feature_idxs = np.random.choice(a=num_features, size=self.max_features, replace=False)

        # Greedy search: find the best feature to be used as discriminant in splitting the node.
        best_feature, best_threshold = self._find_best_feature(X, y, feature_idxs)

        # Split the node using best_feature and best_threshold
        X_column = X[:, best_feature]
        left_child_idxs, right_child_idxs = split_node(X_column, best_threshold)

        # Create child nodes
        left_child_node = self._grow_tree(X[left_child_idxs,:], y[left_child_idxs], depth+1)
        right_child_node = self._grow_tree(X[right_child_idxs,:], y[right_child_idxs], depth+1)

        return DT_Node(feature=best_feature, feat_threshold=best_threshold,
                       left_child=left_child_node, right_child=right_child_node)


    def _find_best_feature(self, X, y, feature_idxs):
        # Evaluate the information gain obtained by using each feature as a discriminant for node split.
        highest_information_gain = -1
        best_feat_idx, best_feat_threshold = None, None

        for i in feature_idxs:
            # Select column i from X, recall X = num_samples x num_features
            X_i = X[:, i]
            thresholds = np.unique(X_i)
            for t in thresholds:
                # Calculate the information gain for feature i with threshold t:
                gain = information_gain(X_i, y, t)

                if gain > highest_information_gain:
                    highest_information_gain = gain
                    best_feat_idx = i
                    best_feat_threshold = t

        return best_feat_idx, best_feat_threshold

    def predict(self, X):
        # Traverse the tree for each row in X
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, parent_node):
        # Check if parent_node is a leaf node
        if parent_node.is_leaf_node():
            return parent_node.leaf_value
        if x[parent_node.feature] <= parent_node.feat_threshold:
            return self._traverse_tree(x, parent_node.left_child)
        return self._traverse_tree(x, parent_node.right_child)

