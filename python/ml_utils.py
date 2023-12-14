import numpy as np
from collections import Counter

def most_common_label(y):
    counter = Counter(y)    # Returns a list of tuples, ea tuple (A,B) where A = label, B = count of label
    # Return the first value of the first tuple. Here we use n=1 so most_common returns only the most common.
    # Each common value generates a list of tuples. So we use [0] to select the first tuple, and [0][0] to
    # retrieve the first value of that tuple (which is the the most common value).
    return counter.most_common(1)[0][0]

def most_common_label_probability(y):
    counter = Counter(y)    # Returns a list of tuples, ea tuple (A,B) where A = label, B = count of label
    # Return the first value of the first tuple. Here we use n=1 so most_common returns only the most common.
    # Each common value generates a list of tuples. So we use [0] to select the first tuple, and [0][0] to
    # retrieve the first value of that tuple (which is the the most common value).
    return counter.most_common(1)[0][0], counter.most_common(1)[0][1]/len(y)


def sqr_euclidean_dist(x1, x2):
    return np.sum((x1 - x2) ** 2)


def euclidean_dist(x1, x2):
    return np.sqrt(sqr_euclidean_dist(x1, x2))


def accuracy(predicted, actual):
    return np.sum([predicted == actual]) / len(actual)


def mse(actual, predicted):
    # Returns mean squared error
    return np.sum((actual - predicted)**2) / len(actual)


def linear_model(X, W, bias):
    return np.dot(X, W) + bias


def sigmoid(v, coeff=1):
    return 1/(1 + np.exp(-v*coeff))


def gaussian_pdf(x, mean, var):
    x_mean = x - mean
    exponent = -0.5 * x_mean.T.dot(np.linalg.inv(var)).dot(x_mean)
    numerator = np.exp(exponent)
    denom = np.sqrt(np.linalg.det(var)) * ((2*np.pi)**(mean.shape[0]/2.0))
    return numerator/denom


def gaussian_univariate_pdf(x, mean, var):
    numerator = np.exp(-((x - mean)**2) / (2*var))
    denom = np.sqrt(var*2*np.pi)
    return numerator/denom