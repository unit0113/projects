import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''# Import MNIST data
mnist = pd.read_csv('Data\mnist_784.csv')

# One hot encoding for labels
def one_hot(label):
    result = np.zeros(10)
    result[int(label)] = 1
    return result

mnist['class'] = mnist['class'].apply(one_hot)

mnist['split'] = np.random.randn(mnist.shape[0], 1)
split_mask = np.random.rand(len(mnist)) <= 0.8
train = mnist[split_mask].copy()
test = mnist[~split_mask].copy()
train.drop('split', axis=1, inplace=True)
test.drop('split', axis=1, inplace=True)

X_train = train.iloc[:,:-1]
y_train = train['class']
X_test = test.iloc[:,:-1]
y_test = test['class']

# Make all of these np arrays rather than dataframes
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()'''

X_train = np.array([[4,7,9,3,1],
                    [7,2,6,8,0],
                    [3,6,8,9,5],
                    [1,0,6,2,5],
                    [0,4,8,4,0],
                    [1,9,0,4,6],
                    [6,9,4,2,3],
                    [8,1,5,4,1],
                    [5,5,1,3,9],
                    [4,7,9,7,3]
                    ])

y = [4,5,7,8,2,6,7,7,5,0]

# One hot encoding for labels
def one_hot(label):
    result = np.zeros(10)
    result[int(label)] = 1
    return result

y_train = np.array([one_hot(lab) for lab in y])


# Mean normalization
#(x-mean)/(max - min)   values end between -0.5 and 0.5, multiply by 2 for -1 to 1




# Initialize NN
class Model:
    def __init__(self, input_size, net_specs, output_size, learning_rate=0.01):
        self.layer_list = [input_size] + net_specs + [output_size]
        self.learning_rate = learning_rate
        
        self.network = []
        for index, layer in enumerate(self.layer_list[:-1]):
            new_layer = self.Layer(layer, self.layer_list[index+1], 'ReLU', self.learning_rate)
            self.network.append(new_layer)


    def set_activation(self, layer, new_activation):
        self.network[layer].activation = new_activation

    
    def f_propogation(self, input): #np.dot(inputs, weights)
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


    class Layer(object):
        """ Individual layer in ANN

        Args:
            n_inputs: Number of inputs (int)
            n_neurons: Number of neurons in layer (int)
            activation: Activation function of the layer. Default is 'sigmoid'. Options: 'sigmoid', 'binary', 'linear', 'tanh', 'ReLU', 'leaky_ReLU', 'soft_max'
        """ 

        def __init__(self, n_inputs, n_neurons, activation='ReLU', learning_rate=0.01):
            self.act_fxns = ['sigmoid', 'binary', 'linear', 'tanh', 'ReLU', 'leaky_ReLU', 'soft_max']
            self.weights = np.random.uniform(-0.5,0.5, (int(n_inputs), int(n_neurons))) # inputs before neurons to avoid need for transposing
            self.biases = np.zeros((int(n_neurons),1))
            self._activation = activation
            self.learning_rate = learning_rate


        @property
        def activation(self):
            return self._activation


        @activation.setter
        def activation(self, new_activation):
            if new_activation not in self.act_fxns:
                raise ValueError('Invalid Activation Function')
            self._activation = new_activation


        def forward(self, input):
            self.input = input
            self.output = np.dot(self.input, self.weights) + self.biases
            match self.activation:
                case 'sigmoid':
                    return self.sigmoid(self.output)
                
                case 'binary':
                    return self.binary(self.output)

                case 'linear':
                    return self.linear(self.output)

                case 'tanh':
                    return self.tanh(self.output)

                case 'ReLU':
                    return self.relu(self.output)

                case 'soft_max':
                    return self.soft_max(self.output)

                case 'leaky_ReLU':
                    return self.leaky_ReLU(self.output)


        def backward(self, output_gradiant):
            pass


        def sigmoid(self, x):
            return np.exp(x - np.max(x)) / (1 + np.exp(x - np.max(x))) # stable version of sigmoid, regular = 1 / (1+np.exp(-x))


        def binary(self, x):
            return np.heaviside(x, 1)


        def linear(self,x):
            return x
        

        def tanh(self, x):
            return 2 * self.sigmoid(2*x) - 1

        
        def soft_max(self, x): # For output layer
            return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


        def relu(self, x):
            return np.maximum(0, x)

        
        def leaky_ReLU(self, x):
            return np.maximum(0.01 * x, x)

        
        def sigmoid_derivative(self, x):
            return self.sigmoid(x) * (1 - self.sigmoid(x))


        def tanh_derivative(self, x):
            return 1 - self.sigmoid(x) ** 2


        def ReLU_derivative(self, x):
            return 0 if x < 0 else 1


        def leaky_ReLU_derivative(self, x):
            return 0.01 if x < 0 else 1





model = Model(X_train[0].shape[0], [10, 5, 10], y_train[0].shape[0], 0.01)

act = model.network[0].activation
print(act)
model.set_activation(0, 'ReLU')
print(model.network[0].activation)