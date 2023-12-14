import numpy as np
import ml_utils

# For details:
# https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html

# Given X independent variables (d-dimension), and y function values, use a linear model to estimate the data scatter
# y = WX + b where W is a d-mensional weight vector, and b is a scalar bias
# Use the RMS (root mean square) loss function and gradient descend to update W.
# W' = W - lr* (dL/dW)
# b' = b - lr& (dL/db)


class LinearRegression:
    def __init__(self):
        self.bias = None
        self.W = None

    def fit(self, X, y, learn_rate=0.01, n_iters=100):
        self.lr = learn_rate
        self.n_iters = n_iters
        n_samples, n_features = X.shape
        print('LinearRegression fitting :')
        print('\tlearn_rate= ', learn_rate)
        print('\tn_iters= ', n_iters)
        print('\tn_samples= ', n_samples)
        print('\tn_features=', n_features)

        # Initialize W and bias
        self.W = np.zeros(n_features)
        self.bias = 0

        # Gradient descend
        for i in range(n_iters):
            y_hat = ml_utils.linear_model(X, self.W, self.bias)
            # Get partial derivative of Loss w.r.t. W
            # Must use the transpose(X) because we want to do the dot-product along axis-1
            # Transpose transforms axis-1 to axis-0.
            dw = np.dot(X.T, (y_hat - y)) / n_samples

            # Get partial derivative of Loss w.r.t. bias
            db = np.sum(y_hat - y) / n_samples

            # Update W and bias
            self.W -= learn_rate * dw
            self.bias -= learn_rate * db

            mse = ml_utils.mse(y, y_hat)

    def predict(self, X):
        return ml_utils.linear_model(X, self.W, self.bias)

