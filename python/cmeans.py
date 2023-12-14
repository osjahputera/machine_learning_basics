import numpy as np
import ml_utils
import matplotlib.pyplot as plt

class CMEANS:
    def __init__(self, n_clusters=3, n_iters=100, convergence_threshold=0.001):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.convergence_threshold = convergence_threshold
        self.cluster_members = None
        self.centroids = None

    def cluster(self, X, display_step=False):
        n_samples, n_features = X.shape
        # Initialize centroids using some samples.
        init_idxs = np.random.choice(a=n_samples, size=self.n_clusters, replace=False)
        self.centroids = np.array(X[init_idxs])

        for iter in range(self.n_iters):
            # Generate clusters from X based on nearest cluster centroid.
            new_clusters = self._find_clusters(X, self.centroids)
            new_centroids = self._get_centroids(X, new_clusters)
            if self._is_converged(new_centroids, self.centroids, self.convergence_threshold):
                self.centroids = new_centroids
                self.cluster_members = new_clusters
                break;
            self.centroids = new_centroids
            self.cluster_members = new_clusters

            if display_step:
                self.plot(X)
        return self.cluster_members

    def _find_clusters(self, X, centroids):
        nearest_centroid_idx = np.array([self._find_nearest_centroid(x, centroids) for x in X])
        clusters = []
        for i in range(centroids.shape[0]):
            cluster_i = np.argwhere(nearest_centroid_idx == i).flatten()
            clusters.append(cluster_i)
        return clusters

    def _find_nearest_centroid(self, x, centroids):
        # x is a single sample with 1 x n_features shape.
        # centroids is a n_clusters x n_features shape (ea centroid is stored in column vector)
        dist_squares = [ml_utils.sqr_euclidean_dist(x, c) for c in centroids]
        return np.argmin(dist_squares)

    def _get_centroids(self, X, clusters):
        # X is n_samples x n_features
        # clusters is n_clusters x [list of members idx in X]
        centroids = [np.mean(X[c], axis=0) for c in clusters]
        return np.array(centroids)

    def _is_converged(self, c1, c2, delta):
        distances = np.zeros(c1.shape[0])
        for idx, centroid1 in enumerate(c1):
            distances[idx] = ml_utils.euclidean_dist(centroid1, c2[idx])
        mean_dist = np.mean(distances)
        print('mean_dist= ', mean_dist)
        return mean_dist <= delta

    def plot(self, X):
        fig, ax = plt.subplots(figsize=(6, 4))

        for i, index in enumerate(self.cluster_members):
            point = X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()




