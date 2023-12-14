import numpy as np
import ml_utils

# For details see:
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html

# In linear regression we have f(X; W, b) = WX + b where W is an 1xD array, X is an NxD array, and b is scalar
# In logistic regression we define h(X; W, b) = 1/[1 + exp(-f(X))]
# Recall the sigmoid function 1/[1 + exp(-kx)] as one of neuron activation function.

# In linear regression we estimate continuous value.
# In logistic regression we estimate probability, recall, the sigmoid function returns value in [0,1]
# Logistic regression predicts probability for each discrete classes or categories.

# We use the cross-entropy cost function instead of RMSE.
# Logistic regression for a binary prediction (returns 1 if a sample is member of this class, 0 else)
# It can be extended to predict membership among n classes. It does it by having n binary regression
# We select the regression with the highest probability as the class assignment.  This is basically
# the SoftMax cost function used in MLP. E.g., MLP with n output neurons predicting n classes is
# one forms of logistic regression.

class LogisticRegression:
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
            z = ml_utils.linear_model(X, self.W, self.bias)
            y_hat = ml_utils.sigmoid(z)

            # The partial derivative of the cross-entropy cost function is surprisingly
            # reduced to the same equation as those found for the linear regression!!

            # Calculate partials w.r.t. W
            dw = np.dot(X.T, (y_hat - y)) / n_samples

            # Calculate partials w.r.t. bias
            db = np.sum(y_hat - y) / n_samples

            # Update W and bias
            self.W -= self.lr * dw
            self.bias -= self.lr * db

    def predict_probability(self, X):
        # It is useful to have a predict function that returns probability
        # so this class can be used in multi-class logistic regression
        z = ml_utils.linear_model(X, self.W, self.bias)
        return ml_utils.sigmoid(z)

    def predict(self, X):
        y_probability = self.predict_probability(X)
        class_prediction = [1 if p > 0.5 else 0 for p in y_probability]
        return class_prediction
