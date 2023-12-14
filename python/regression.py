import numpy as np
import ml_utils

# We rewrite the LinearRegression and LogisticRegression using a base class Regression


class Regression:
    def __init__(self, lr=0.001, iters=1000):
        self.lr = lr
        self.iters = iters
        self.W = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.bias = 0
        print('LogisticRegression fitting :')
        print('\tlearn_rate= ', self.lr)
        print('\titers= ', self.iters)
        print('\tn_samples= ', n_samples)
        print('\tn_features=', n_features)

        for i in range(self.iters):
            y_hat = self._approximate(X, self.W, self.bias)

            # The partial derivative of the cross-entropy cost function is surprisingly
            # reduced to the same equation as those found for the linear regression!!

            # Calculate partials w.r.t. W
            dw = np.dot(X.T, (y_hat - y)) / n_samples

            # Calculate partials w.r.t. bias
            db = np.sum(y_hat - y) / n_samples

            # Update W and bias
            self.W -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return self._predict(X, self.W, self.bias)

    def predict_probability(self, X):
        return self._approximate(X, self.W, self.bias)

    def _approximate(self, X, W, b):
        raise NotImplementedError()

    def _predict(self, X, W, b):
        raise NotImplementedError()


class LinearRegression(Regression):
    def _approximate(self, X, W, b):
        return ml_utils.linear_model(X, W, b)

    def _predict(self, X, W, b):
        return self._approximate(X, W, b)


class LogisticRegression(Regression):
    def _approximate(self, X, W, b):
        linear_model = ml_utils.linear_model(X, W, b)
        return ml_utils.sigmoid(linear_model)

    def _predict(self, X, W, b):
        y_probability = self._approximate(X, W, b)
        class_prediction = [1 if p > 0.5 else 0 for p in y_probability]
        return class_prediction
