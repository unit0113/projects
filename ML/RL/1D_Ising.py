import numpy as np
import torch
from matplotlib import pyplot as plt


def init_grid(size=(10,)):
    grid = torch.randn(*size)
    grid[grid > 0] = 1
    grid[grid <= 0] = 0
    grid = grid.byte() # Converts floats to binary
    return grid


def get_reward(s, a):
    # Rewards generated based on neighbor's states
    r = -1
    for i in s:
        if i == a:
            r += 0.9
    r *= 2.
    return r


def gen_params(N, size):
    # Generates list of paramater vectors for NN
    ret = []
    for _ in range(N):
        vec = torch.randn(size) / 10.
        vec.requires_grad = True
        ret.append(vec)
    return ret


def qfunc(s,theta,layers=[(4,20),(20,2)], afn=torch.tanh):
    l1n = layers[0] 
    l1s = np.prod(l1n) # Takes first tuple in layers and multiplies those numbers to get the subset of the theta vector to use as the first layer
    theta_1 = theta[0:l1s].reshape(l1n) # Reshapes theta vector subset into a matrix for use as first layer in NN
    l2n = layers[1]
    l2s = np.prod(l2n)
    theta_2 = theta[l1s:l2s+l1s].reshape(l2n)
    bias = torch.ones((1,theta_1.shape[1]))
    l1 = s @ theta_1 + bias # First layer computation
    l1 = torch.nn.functional.elu(l1)
    l2 = afn(l1 @ theta_2) # Default tanh activation fxn
    return l2.flatten()


def get_substate(b): # converts binary number to one-hot encoded vector
    s = torch.zeros(2) 
    if b > 0: # if up, else down
        s[1] = 1
    else:
        s[0] = 1
    return s


def joint_state(s): # s[0] = left neighbor, s[1] = right
    s1_ = get_substate(s[0]) # get action vectors for each neighbor
    s2_ = get_substate(s[1])
    ret = (s1_.reshape(2,1) @ s2_.reshape(1,2)).flatten() # Create joint action space using out-product, then flatten
    return ret


def main():
    plt.figure(figsize=(8,5))
    size = (20,)
    hid_layer = 20
    params = gen_params(size[0],4*hid_layer+hid_layer*2) # Generates list of param vectors that will parameterize the Q fxn
    grid = init_grid(size=size)
    grid_ = grid.clone() # Needed for training
    plt.imshow(np.expand_dims(grid,0))
    plt.show()
    
    epochs = 200
    lr = 0.001
    losses = [[] for i in range(size[0])] # Tracking losses for each agent
    for _ in range(epochs):
        for j in range(size[0]): # iterate through each agent
            l = j - 1 if j - 1 >= 0 else size[0] - 1 # Gets left neighbor
            r = j + 1 if j + 1 < size[0] else 0 # Gets right neighbor
            state_ = grid[[l, r]] # Spins of left and right neighbors
            state = joint_state(state_) # one hot joint action vector
            qvals = qfunc(state.float().detach(), params[j], layers=[(4, hid_layer), (hid_layer, 2)])
            qmax = torch.argmax(qvals, dim=0).detach().item() # Greedy policy
            action = int(qmax)
            grid_[j] = action # Take action in grid copy, applied to main grid once all actions taken
            reward = get_reward(state_.detach(),action)
            with torch.no_grad(): # Tgt value is the Q value vector with the Q value associated with the action taken replaced with the observed reward
                target = qvals.clone()
                target[action] = reward
            loss = torch.sum(torch.pow(qvals - target,2))
            losses[j].append(loss.detach().numpy())
            loss.backward()
            with torch.no_grad(): # Manual gradient descent
                params[j] = params[j] - lr * params[j].grad
            params[j].requires_grad = True
        with torch.no_grad(): # Copies temp grid into primary grid
            grid.data = grid_.data

    fig, ax = plt.subplots(2,1)
    for i in range(size[0]):
        ax[0].scatter(np.arange(len(losses[i])),losses[i])
    ax[1].imshow(np.expand_dims(grid,0))
    plt.show()


if __name__ == '__main__':
    main()