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

model_dict = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}

def retrieve(model_dict):
    return model_dict['w1'], model_dict['w2'], model_dict['b1'], model_dict['b2']

def forward(x, model_dict):
    w1, w2, b1, b2 = retrieve(model_dict)
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)

    return np.exp(a2) / np.sum(np.exp(a2), axis=1, keepdims=True)  # Softmax

def log_loss(y_pred, y):
    m = np.zeros(2000)
    for i, label in enumerate(y):
        m[i] = y_pred[i][label]

    log_prob = -np.log(m)
    return np.sum(log_prob)

def reg_loss(model_dict):
    w1, w2, _, _ = retrieve(model_dict)
    return reg_lambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

def loss(y_pred, y, model_dict):
    return log_loss(y_pred, y) + reg_loss(model_dict)

def predict(x, model_dict):
    w1, w2, b1, b2 = retrieve(model_dict)
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)

    softmax = np.exp(a2) / np.sum(np.exp(a2), axis=1, keepdims=True)
    return np.argmax(softmax, axis=1)

    