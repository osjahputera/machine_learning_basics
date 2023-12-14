import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.iters = iters
        self.W = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # The prediction output of SVM is given in either +1 or -1 value to label 2 classes.
        # We must insure that y has the same value choices as well.
        y_train = np.where(y >= 0, 1, -1)
        self.W = np.zeros(n_features)
        self.bias = 0

        for iter in range(self.iters):
            correct_count = 0
            for i, x_i in enumerate(X):
                f_i = self._linear_model(x_i)
                y_f_i = f_i * y_train[i]

                # Gradients
                if y_f_i >= 1:
                    dw = 2 * self.lambda_param * self.W
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.W - (x_i * y_train[i])
                    db = y_train[i]

                # Update
                self.W -= self.lr*dw
                self.bias -= self.lr*db

                if y_f_i >= 1:
                    correct_count += 1
            if iter % 20 == 0:
                print('Iter=', iter, ' score=', correct_count/n_samples)
            if correct_count == n_samples:
                print('Convergence reached at iter=', iter)
                break

    def predict(self, X):
        linear_model = self._linear_model(X)
        # Returns predicted label as +1 or -1
        return np.sign(linear_model)

    def _linear_model(self, X):
        return np.dot(X, self.W) - self.bias