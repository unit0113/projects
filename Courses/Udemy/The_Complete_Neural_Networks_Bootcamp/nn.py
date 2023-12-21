import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

X, y = sklearn.datasets.make_moons(2000, noise = 0.15)
#plt.scatter(X[:,0], X[:,1], c=y)
#plt.show()

# Hyperparameters
input_neurons = 2
output_neurons = 2
num_samples = X.shape[0]
learning_rate = 0.001
reg_lambda = 0.01
epochs = 1500


def retrieve(model_dict):
    return model_dict['w1'], model_dict['w2'], model_dict['b1'], model_dict['b2']

def forward(x, model_dict):
    w1, w2, b1, b2 = retrieve(model_dict)
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)
    softmax = np.exp(a2) / np.sum(np.exp(a2), axis=1, keepdims=True)

    return z1, a1, softmax

def loss(softmax, y, model_dict):
    W1, b1, W2, b2 = retrieve(model_dict)
    m = np.zeros(len(y))
    for i,correct_index in enumerate(y):
        predicted = softmax[i][correct_index]
        m[i] = predicted
    log_prob = -np.log(m)
    loss = np.sum(log_prob)
    reg_loss = reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    loss+= reg_loss
    return float(loss / y.shape[0])

def predict(x, model_dict):
    w1, w2, b1, b2 = retrieve(model_dict)
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)

    softmax = np.exp(a2) / np.sum(np.exp(a2), axis=1, keepdims=True)
    return np.argmax(softmax, axis=1)

def backward(x, y, model_dict, epochs):
    for i in range(epochs):
        w1, w2, b1, b2 = retrieve(model_dict)
        z1, a1, output = forward(x, model_dict)
        delta3 = np.copy(output)
        delta3[range(y.shape[0]), y] -= 1   # subtract 1 from positive labels
        dw2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(w2.T) * (1 - np.power(np.tanh(z1), 2))
        dw1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # regularization
        dw2 += reg_lambda * np.sum(w2)
        dw1 += reg_lambda * np.sum(w1)

        # Update weights
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2

        # Update bias
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2

        # Update model
        model_dict = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}

        # Print loss
        if i % 50 == 0:
            print(f"Epoch {i} loss: {loss(output, y, model_dict)}")

    return model_dict

def init_network(input_dim, hidden_dim, output_dim):
    w1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    b1 = np.random.randn(1, hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    b2 = np.random.randn(1, output_dim)

    model_dict = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
    return model_dict


model_dict = init_network(input_neurons, 3, output_neurons)
model = backward(X, y, model_dict, epochs)