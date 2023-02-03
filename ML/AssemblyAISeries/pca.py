import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calc Covariance
        cov = np.cov(X.T)       # T because functions requires data as columns

        # Calc Eigenvectors and Eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)          # Results in column vectors
        eigenvectors = eigenvectors.T

        # Sort Eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]


    def transform(self, X):
        # Mean centering
        X = X - self.mean

        # Project data
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()