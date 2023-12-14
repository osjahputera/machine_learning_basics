import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.transform_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.means = None
        self.covariance = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.n_components = min(self.n_components, n_features) if not self.n_components else n_features
        self.means = np.mean(X, axis=0)
        # Here we translated  our data by the means (so the new means will be at the origin)
        # and calculate the covariance matrix from the translated data
        X_translated = X - self.means
        self.covariance = np.cov(X_translated, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance)
        # np.linalg.eig returns eigenvectors as column vectors where eigencov[:,i]
        # is the eigen vector that corresponds to the i-th eigen value.
        # need to have it as row vectors so the eigen vector is in eigencov[i,:]
        self.eigenvectors = self.eigenvectors.T

        # Order eigenvalues in decreasing order
        sorted_eigenvalues_idxs = np.argsort(self.eigenvalues)[::-1]

        # Create the transform matrix T where T[:,i] is the i-th eigen vector
        self.transform_matrix = self.eigenvectors[sorted_eigenvalues_idxs[0:self.n_components]]
        self.transform_matrix = self.transform_matrix.T

    def transform(self, X):
        X_trans = np.dot(X - self.means, self.transform_matrix)
        return X_trans

    def inverse_transform(self, X):
        X_inv = np.dot(X, self.transform_matrix.T) + self.means
        return X_inv
