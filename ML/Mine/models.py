import layers
import optimizers
import loss_functions


class Sequential:
    def __init__(self, *layers: list[layers.Layer], optimizer: optimizers.Optimizer = optimizers.Adam(), loss_function: str = 'mse') -> None:
        self.layers = layers
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, X, y) -> None:
        pass

    def predict(self, X):
        pass

    @property
    def summary(self) -> None:
        pass

    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass