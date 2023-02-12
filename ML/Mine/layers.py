import numpy as np
import activation_functions
from abc import ABC, abstractmethod, abstractproperty


class Layer(ABC):
    @abstractmethod
    def forward(self, X) -> None:
        pass

    @abstractmethod
    def backwards(self, dvalues) -> None:
        pass

    @abstractmethod
    def get_parameters(self) -> list[np.array]:
        pass
    
    @abstractmethod
    def set_parameters(self, *args) -> None:
        pass

    @abstractproperty
    def summary(self) -> str:
        pass


class Dense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int, activation: str = 'relu', weight_regularizer_l1: float = 0, weight_regularizer_l2: float = 0, bias_regularizer_l1: float = 0, bias_regularizer_l2: float = 0) -> None:
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 weight gradients
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 weights gradients
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 biases gradients
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 biases gradients
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

    @property
    def summary(self) -> str:
        pass