import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:
    def __init__(self, K=5, max_iters=1000, plot_steps=False) -> None:
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = self.X.shape

        # Initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization loop
        for _ in range(self.max_iters):
            # Assign samples to closest centroid
            self.clusters = self._create_clusters()

            if self.plot_steps:
                self.plot()

            # Calc new centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids()

            if self._is_converged(centroids_old):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as index of their clusters
        return self._get_cluster_labels()


    def _create_clusters(self):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)

        return clusters

    def _closest_centroid(self, sample):
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids

    def _is_converged(self, centroids_old):
        distances = [euclidean_distance(centroids_old[i], self.centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

    



# Testing
if __name__ == "__main__":
    np.random.seed(42)
    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()