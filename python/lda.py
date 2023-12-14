import numpy as np

class LDA:
    def __init__(self, k=2):
        self.k = k
        self.linear_discriminants = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        unique_class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)

        # Within class scatter matrix
        S_W = np.zeros((n_features, n_features))
        # Between class scatter matrix
        S_B = np.zeros((n_features, n_features))
        for class_i in unique_class_labels:
            X_label = X[y == class_i,:]
            n_class_i = len(X_label)
            mean_class_i = np.mean(X_label, axis=0)
            S_W += (X_label - mean_class_i).T.dot(X_label - mean_class_i)
            mean_diff = mean_class_i - mean_overall
            S_B += n_class_i * (mean_diff).dot(mean_diff.T)

        # A = S_B/S_W = S_B*inv(S_W)
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # np.linalg.eig returns eigenvectors as column vectors where eigencov[:,i]
        # is the eigen vector that corresponds to the i-th eigen value.
        # need to have it as row vectors so the eigen vector is in eigencov[i,:]
        eigenvectors = eigenvectors.T

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        eigval_sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[eigval_sorted_idx]
        self.eigenvectors = eigenvectors[eigval_sorted_idx,:]
