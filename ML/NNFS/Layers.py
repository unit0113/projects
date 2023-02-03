import numpy as np
from scipy import signal


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2


    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    
    def get_parameters(self):
        return self.weights, self.biases

    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Layer_Dropout:
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate


    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask


    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs


class Activation_ReLU:
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)


    def backward(self, dvalues):
        # Since we need to modify original variable, let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


class Activation_Softmax:
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:
    def forward(self, inputs, training):
        # Save input and calculate/save output of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))


    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output


    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs


    def backward(self, dvalues):
        # Derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()


    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


class Convolutional2D:
    def __init__(self, *, input_shape, kernel_size, filters):
        input_depth, input_height, input_width = input_shape
        self.filters = filters
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (filters, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        

    def backward(self, dvalues):
        kernels_gradient = np.zeros(self.kernels_shape)
        self.dinputs = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], dvalues[i], "valid")
                self.dinputs[j] += signal.convolve2d(dvalues[i], self.kernels[i, j], "full")

        self.kernels -= kernels_gradient
        self.biases -= dvalues


class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs):
        return np.reshape(inputs, self.output_shape)

    def backward(self, dvalues):
        return np.reshape(dvalues, self.input_shape)