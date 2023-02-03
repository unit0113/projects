import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import MNIST data
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
y_test = y_test.to_numpy()

# Initialize NN
w_i_h = np.random.uniform(-0.5,0.5, (40, 784)) # weights, input to hidden
w_h_o = np.random.uniform(-0.5,0.5, (10, 40))  # weights, hidden to output
b_i_h = np.zeros((40,1))                      # bias, input to hidden
b_h_o = np.zeros((10,1))                      # bias, hidden to output

# Convolution


# Pooling


# Flattening


# Training cycles
learning_rate = 0.01
epochs = 100
for epoch in range(epochs):
    nr_correct = 0
    for img, lab in zip(X_train, y_train):
        # Reshape img, lab for matrix multiplication
        img.shape += (1,)
        lab_new = lab.reshape(-1,1)
        
        # Forward propogation, input to hidden
        h_pre = b_i_h + w_i_h @ img     # @ is matrix multiplication
        h = 1 / (1 + np.exp(-h_pre))    # np.exp is matrix exponential, e^x for each element        subtract max to ovoid overflow

        # Forward propogation, hidden to output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))    # Used for normalization (sigmoid function in this case)

        # Cost function
        e = 1 / len(o) * np.sum((o - lab_new) ** 2, axis = 0)   # Mean squared error
        nr_correct += int(np.argmax(o) == np.argmax(lab_new))   # np.argmax Returns the indices of the maximum values along an axis

        # Backpropogation, output to hidden
        delta_o = o - lab_new   # e can be used to calc delta_o. Normally the derivative of the cost function

        w_h_o += -learning_rate * delta_o @ np.transpose(h)
        b_h_o += -learning_rate * delta_o

        # Back propogation, hidden to input
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learning_rate * delta_h @ np.transpose(img)
        b_i_h += -learning_rate * delta_h

    # Display accuracy for this epoch
    print(f"Accuracy of Epoch {int(epoch + 1)}: {round((nr_correct / X_train.shape[0]) * 100, 2)}%")

# Testing
nr_correct = 0
for img, lab in zip(X_test, y_test):
    # Reshape img, lab for matrix multiplication
    img.shape += (1,)
    lab_new = lab.reshape(-1,1) # should have used np.expand_dims(lab_new, axis=0)
    
    # Forward propogation, input to hidden
    h_pre = b_i_h + w_i_h @ img     # @ is matrix multiplication
    h = 1 / (1 + np.exp(-h_pre))    # np.exp is matrix exponential, e^x for each element

    # Forward propogation, hidden to output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))    # Used for normalization (sigmoid function in this case)

    nr_correct += int(np.argmax(o) == np.argmax(lab_new))

# Display accuracy for this epoch
print(f"Accuracy of testing: {round((nr_correct / X_train.shape[0]) * 100, 2)}%")



# Show results
while True:
    output_str = f'Enter a number (0 - {len(X_test)}): '
    index = int(input(output_str))
    img = X_test[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"I think its a {o.argmax()}")
    plt.show()