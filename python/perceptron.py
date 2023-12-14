import numpy as np
import ml_utils

# Must be used with linearly separable data!!

class Perceptron:
    def __init__(self, lr=0.001, iters=1000):
        self.lr = lr
        self.iters = iters
        self.activation_func = self._unit_step_function
        self.W = None
        self.bias = None

    # We can try using a different activation functions like the sigmoid.
    # If we use the sigmoid, the update equation for W and bias must be changed
    # following the partial derivative of the sigmoid. We essentially enter
    # the gradient descent zone when the activation function is differentiable.

    def _unit_step_function(self, x):
        # Wants to return 1 if x >= 0, else 0
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features, dtype=np.float64)
        self.bias = 0

        # y must contains 0 or 1 value!
        y_scaled = np.array([1 if y_val > 0 else 0 for y_val in y])

        for i in range(self.iters):
            correct_count = 0;
            for idx, x_idx in enumerate(X):
                linear_model = ml_utils.linear_model(x_idx, self.W, self.bias)
                y_predicted = self.activation_func(linear_model)

                # Update W and bias
                update = self.lr * (y_predicted - y_scaled[idx])
                self.W -= x_idx * update
                self.bias -= update

                if y_predicted == y_scaled[idx]:
                    correct_count += 1
            if correct_count == n_samples:
                print("Convergence at iter=", i)
                break
            else:
                print("Accuracy at iter=", i, " : ", float(correct_count)/n_samples)

    def predict(self, X):
        linear_model = ml_utils.linear_model(X, self.W, self.bias)
        activation = self.activation_func(linear_model)
        return activation