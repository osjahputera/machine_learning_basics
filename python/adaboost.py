import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.feature_threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:,self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.feature_threshold] = -1
        else:
            predictions[X_column > self.feature_threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, n_classifiers=5):
        self.n_classifiers = n_classifiers
        self.classifiers = []


    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Normalize the labels y to +1 or -1
        y_min = np.amin(y)
        y_max = np.amax(y)
        y_med = (y_max + y_min)/2.0
        y_normalized = np.ones(n_samples)
        y_normalized[y > y_med] = 1
        y_normalized[y < y_med] = -1

        # Init weights for samples
        weights = np.full(n_samples, 1/n_samples)

        EPS = 1e-10

        for _ in range(self.n_classifiers):
            # Greedy search of feature that works best with this classifier.
            classifier = DecisionStump()
            min_error = float('inf')
            for fidx in range(n_features):
                X_column = X[:, fidx]
                # Get all unique values of feature-fidx, these will become candidate thresholds.
                thresholds = np.unique(X_column)

                for thr in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < thr] = -1
                    misclassified_weights = weights[y != predictions]
                    error = np.sum(misclassified_weights)

                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        min_error = error
                        classifier.polarity = polarity
                        classifier.feature_idx = fidx
                        classifier.feature_threshold = thr
            classifier.alpha = 0.5 * np.log((1 - min_error + EPS)/ (min_error + EPS))
            predictions = classifier.predict(X)
            weights *= np.exp(-classifier.alpha * y_normalized * predictions)
            weights /= np.sum(weights)
            self.classifiers.append(classifier)

    def predict(self, X):
        # This will produce a list of n_classifiers X n_samples.
        predictions = [clf.alpha * clf.predict(X) for clf in self.classifiers]

        # The prediction for each sample in X is given by the sum of alpha_i * predict_i
        # where alpha_i and predict_i are the alpha and prediction result of classifier-i
        y_predicted = np.sum(predictions, axis=0)
        # Our predictions should contain the values of +1 or -1
        y_predicted = np.sign(y_predicted)
        return y_predicted