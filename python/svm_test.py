from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from svm import SVM

X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)

print('X.shape=', X.shape)
print('y.shape=', y.shape)

def get_hyperplane_param(svm_model, x0, offset):
    return (-x0 * svm_model.W[0] + svm_model.bias + offset) / svm_model.W[1]

def visualize_hyperplane(svm_model, X, y):
    x0_1 = np.amin(X[:,0])
    x0_2 = np.amax(X[:,0])

    x1_1_center = get_hyperplane_param(svm_model, x0_1, 0)
    x1_2_center = get_hyperplane_param(svm_model, x0_2, 0)

    x1_1_up = get_hyperplane_param(svm_model, x0_1, 1.0)
    x1_2_up = get_hyperplane_param(svm_model, x0_2, 1.0)

    x1_1_down = get_hyperplane_param(svm_model, x0_1, -1.0)
    x1_2_down = get_hyperplane_param(svm_model, x0_2, -1.0)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X[:,0], X[:,1], c=y, marker='o')

    ax.plot([x0_1, x0_2], [x1_1_center, x1_2_center], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_up, x1_2_up], 'k')
    ax.plot([x0_1, x0_2], [x1_1_down, x1_2_down], 'r')

    x1_min = np.amin(X[:,1])
    x1_max = np.amax(X[:,1])
    ax.set_ylim([x1_min-2, x1_max+2])

    plt.show()

lr = 0.001
lambda_param = 0.01
iters = 1000

S = SVM(lr, lambda_param, iters)
S.fit(X, y)
print('W=', S.W)
print('b=', S.bias)
visualize_hyperplane(S, X, y)