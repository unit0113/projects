import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


class LogisticRegression:
    def __init__(self, lr = 0.01, n_iters = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))      # Stable sigmoid
        #return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_predications = np.dot(X, self.weights) + self.bias
            predications = self._sigmoid(linear_predications)

            dw = np.dot(X.T, (predications - y)) / n_samples
            db = np.sum(predications - y) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_predications = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_predications)
        return [0 if y <= 0.5 else 1 for y in y_pred]


def accuracy(y_test, y_pred):
    return np.sum(y_pred == y_test) / len(y_test)


if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(accuracy(y_test, y_pred))