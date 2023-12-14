from sklearn.datasets import make_blobs
from cmeans import CMEANS
import numpy as np

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = CMEANS(n_clusters=3, n_iters=100, convergence_threshold=0.001)
cluster_members = k.cluster(X, display_step=True)

k.plot(X)