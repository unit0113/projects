import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
#plt.show()




inputs = np.array([[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]])
weights = np.array([[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([2, 3, 0.5])
weights2 = np.array([[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]])
biases2 = np.array([-1, 2, -0.5])


layer1_out = np.dot(inputs, weights.T) + biases
layer2_out = np.dot(layer1_out, weights2.T) + biases2
#print(layer2_out)




class Layer:
    def __init__(self, activation):
        self.act_fxns = ['sigmoid', 'binary', 'linear', 'tanh', 'ReLU', 'leaky_ReLU', 'soft_max', 'softplus', 'softminus']
        self._activation = activation

    @property
    def activation(self):
        return self._activation


    @activation.setter
    def activation(self, new_activation):
        if new_activation not in self.act_fxns:
            raise ValueError('Invalid Activation Function')
        self._activation = new_activation


    def activate(self, x):
        match self.activation:
            case 'sigmoid':                                                     # Mostly replaced by ReLU for hidden layers, still good for classification output, specifically for non-exclusive classivication
                return np.exp(x - np.max(x)) / (1 + np.exp(x - np.max(x)))      # stable version of sigmoid, regular = 1 / (1+np.exp(-x))
            
            case 'binary':                                                      # Used for output layer on regression
                return np.heaviside(x, 1)

            case 'linear':
                return x

            case 'tanh':
                return 2 * self.sigmoid(2*x) - 1

            case 'ReLU':                                                        # Most common hidden layer function
                return np.maximum(0, x)

            case 'soft_max':                                                    # For output layer on exclusive classification
                exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_values / np.sum(exp_values, axis=1, keepdims=True)        # For batches

            case 'leaky_ReLU':
                return np.maximum(0.01 * x, x)

            case 'softplus':
                return np.log(1 + np.exp(x))

            case 'softminus':
                return x - np.log(1 + np.exp(x))

            case 'swish':
                return x / (1 + np.log(-x))

            case 'ELiSH':
                if x > 0:
                    return x / (1 + np.log(-x))
                else:
                    return (np.log(x) -1) / (1 + np.log(-x))

            case 'HardTanH':
                if x < -1:
                    return -1
                elif x > 1:
                    return 1
                else:
                    return x

            case 'TanhRE':
                if x >= 0:
                    return x
                else:
                    return (np.log(x) - np.log(-x)) / (np.log(x) - np.log(-x))

            case 'ELU':
                if x > 0:
                    return x
                else:
                    return np.log(x) - 1

    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def tanh_derivative(self, x):
        return 1 - self.sigmoid(x) ** 2


    def ReLU_derivative(self, x):
        return 0 if x < 0 else 1


    def leaky_ReLU_derivative(self, x):
        return 0.01 if x < 0 else 1


    def soft_plus_derivative(self, x):
        return 1 / (1 + np.exp(-x))


    def soft_minus_derivative(self, x):
        return 1 - 1 / (1 + np.exp(-x))


class Layer_Dense(Layer):
    """ Individual dense layer in ANN

    Args:
        n_inputs: Number of inputs (int)
        n_neurons: Number of neurons in layer (int)
        activation: Activation function of the layer. Default is 'sigmoid'. Options: 'sigmoid', 'binary', 'linear', 'tanh', 'ReLU', 'leaky_ReLU', 'soft_max'
    """ 

    def __init__(self, n_inputs, n_neurons, activation='ReLU'):
        super().__init__(activation)
        self.weights = np.random.uniform(-0.5,0.5, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    
    def forward(self, inputs):
        self.output = self.activate(np.dot(inputs, self.weights) + self.biases)
        return self.output

# Common loss class
class Loss:
# Calculates the data and regularization losses
# given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # For categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        # For one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1
            )
        # Losses
        return -np.log(correct_confidences)


















class Model:
    def __init__(self, input_size, net_specs, output_size, learning_rate=0.01):
        self.layer_list = [input_size] + net_specs + [output_size]
        self.learning_rate = learning_rate
        
        self.network = []
        for index, layer in enumerate(self.layer_list[:-1]):
            new_layer = Layer_Dense(layer, self.layer_list[index+1], 'ReLU')
            self.network.append(new_layer)


    @property
    def learning_rate(self):
        return self.learning_rate


    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        if new_learning_rate is not type(float):
            raise ValueError('Invalid Activation Function')
        self._learning_rate = new_learning_rate



    def set_activation(self, layer, new_activation):
        self.network[layer].activation = new_activation

    
    def f_propogation(self, input):
        running_input = input
        for layer in self.network:
            running_input = layer.forward(running_input) 

        return running_input


    def loss(self, output, label):  # Categorical Cross-Entropy, good for classification with one-hot
        num_samples = len(output)
        output_clipped = np.clip(output, 1e-7, 1-1e-7)

        if len(label.shape) == 1:     # If categorical labels
            correct_confidences = output_clipped[range(num_samples), label]
        
        else:   # for one-hot
            correct_confidences = np.sum(output_clipped * label, axis=1)
        
        return -np.log(correct_confidences)

        # Normal equation? good for num_features < 10000? doesn't need to iterate
        #(X * X.T)^-1 * X.T * y

    
    def b_propogation(self, loss):
        gradient = loss
        for layer in reversed(self.network):
            gradient = layer.backward(gradient)


    def train(self, X_train, y_train, epochs=100):
        for epoch in epochs:
            num_correct = 0
            precision_counter = np.zeros(10)
            for X, y in zip(X_train, y_train):
                output = self.f_propogation(X)
                num_correct += int(np.argmax(output) == np.argmax(y))
                precision_counter[np.argmax(output)] += 1
                loss = self.loss(output, y)
                self.b_propogation(loss)
            
            print(f"Epoch {int(epoch + 1)}:\nAccuracy: {round((num_correct / X_train.shape[0]) * 100, 2)}%\nPrecision: {np.divide(precision_counter, y_train.sum(axis=0))}")


    def test():
        pass


    def dropout():
        pass


    def hypertune(): # tune learning rate, network size/shape, activation functions
        pass




X, y = spiral_data(samples=100, classes=3)
d = Layer_Dense(2, 3)
d.activation = 'ReLU'
d.forward(X)
d2 = Layer_Dense(3,3)
d2.activation = 'soft_max'
d2.forward(d.output)
loss_fxn = Loss_CategoricalCrossentropy()
loss = loss_fxn.calculate(d2.output, y)
print(loss)