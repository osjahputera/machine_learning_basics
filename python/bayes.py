import numpy as np

import ml_utils


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Init the mean and std for each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_class = X[c == y]
            self.mean[idx,:] = np.mean(X_class, axis=0)
            self.var[idx,:] = np.var(X_class, axis=0)
            self.prior[idx] = len(X_class) / float(n_samples)

    def predict(self, X):
        predicted = [self._predict(x) for x in X]
        return predicted

    def _predict(self, x):
        posteriors = []
        evidence = 0
        for idx, c in enumerate(self._classes):
            log_prior = np.log(self.prior[idx])
            # Because we assume all are independent variables (covariance will be a diagonal matrix)
            # Then we can calculate the likelihood as the product of univariate likelihood for each variable.
            # By taking log, we can turn the product into a sum. If the covariance is not diagonal, we
            # need to call the multivariate gaussian pdf.
            log_likelihood = np.sum(np.log(ml_utils.gaussian_univariate_pdf(x, self.mean[idx], self.var[idx])))
            posteriors.append(log_prior + log_likelihood)
        max_idx = np.argmax(posteriors)
        return self._classes[max_idx]




