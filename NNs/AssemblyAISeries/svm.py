import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, lr = 0.001, lambda_param=0.01, n_iters=1000) -> None:
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = 0
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                condition = y_[index] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[index]))
                    self.bias -= self.lr * y_[index]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)


def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


def visualize_svm():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.weights, clf.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.weights, clf.bias, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.weights, clf.bias, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.weights, clf.bias, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.weights, clf.bias, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.weights, clf.bias, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    print("SVM classification accuracy", accuracy(y_test, predictions))

    visualize_svm()