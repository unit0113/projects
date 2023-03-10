import torch
import numpy as np
from matplotlib import pyplot as plt
import gym
from collections import deque
from random import shuffle





def update_dist(r, support, probs, lim=(-10.,10.), gamma=0.8):
    nsup = probs.shape[0]
    vmin, vmax = lim[0], lim[1]
    dz = (vmax - vmin) / (nsup - 1.) # Support spacing value
    bj = np.round((r - vmin) / dz) # Index of observed reward
    bj = int(np.clip(bj, 0, nsup - 1)) # Turns float into valid index
    m = probs.clone()
    j = 1
    for i in range(bj, 1, -1): # Go left and steal portion of distribution
        m[i] += np.power(gamma, j) * m[i - 1]
        j += 1
    j = 1
    for i in range(bj,nsup-1,1): # Go right and steal portion of distribution
        m[i] += np.power(gamma, j) * m[i + 1]
        j += 1
    m /= m.sum() # Ensure sums to 1
    return m


def dist_dqn(x, theta, aspace=3): # x is input state vector, theta is the parameter vector
    dim0, dim1, dim2, dim3 = 128, 100, 25, 51 #Layer demensions for unpacking
    t1 = dim0 * dim1
    t2 = dim2 * dim1
    theta1 = theta[0:t1].reshape(dim0, dim1) # Unpack first portion of theta
    theta2 = theta[t1:t1 + t2].reshape(dim1, dim2)
    l1 = x @ theta1
    l1 = torch.selu(l1)
    l2 = l1 @ theta2
    l2 = torch.selu(l2)
    l3 = []
    for i in range(aspace): # Generate action-value distributions
        step = dim2*dim3
        theta5_dim = t1 + t2 + i * step
        theta5 = theta[theta5_dim:theta5_dim+step].reshape(dim2,dim3)
        l3_ = l2 @ theta5
        l3.append(l3_)
    l3 = torch.stack(l3,dim=1)
    l3 = torch.nn.functional.softmax(l3,dim=2)
    return l3.squeeze()


def get_target_dist(dist_batch, action_batch, reward_batch, support, lim=(-10,10), gamma=0.8):
    nsup = support.shape[0]
    vmin, vmax = lim[0], lim[1]
    dz = (vmax - vmin) / (nsup - 1.)
    target_dist_batch = dist_batch.clone()
    for i in range(dist_batch.shape[0]): # Loop through batch dimension
        dist_full = dist_batch[i]
        action = int(action_batch[i].item())
        dist = dist_full[action]
        r = reward_batch[i]
        if r != -1: # If terminal state and target is a degenerate distribution at the reward value
            target_dist = torch.zeros(nsup)
            bj = np.round((r - vmin) / dz)
            bj = int(np.clip(bj, 0, nsup-1))
            target_dist[bj] = 1.
        else: # Non-terminal, target distribution is a bayesian update of the prior reward
            target_dist = update_dist(r,support,dist,lim=lim,gamma=gamma)
        target_dist_batch[i,action,:] = target_dist # Only chance distribution for taken action
        
    return target_dist_batch


def lossfn(x, y): # Cross-entrophy
    loss = torch.Tensor([0.])
    loss.requires_grad=True
    for i in range(x.shape[0]): # Loop through batch dimension
        loss_ = -1 *  torch.log(x[i].flatten(start_dim=0)) @ y[i].flatten(start_dim=0) # Inner product
        loss = loss + loss_
    return loss


def preproc_state(state):
    p_state = torch.from_numpy(state).unsqueeze(dim=0).float()
    p_state = torch.nn.functional.normalize(p_state, dim=1) # Normalize state values between 0-1
    return p_state

def get_action(dist,support):
    actions = []
    for b in range(dist.shape[0]): # Loop through batch dimension
        expectations = [support @ dist[b, a, :] for a in range(dist.shape[1])] # Computes expectation values for each action-value distribution
        action = int(np.argmax(expectations)) # Computes action with the highest expectation value
        actions.append(action)
    actions = torch.Tensor(actions).int()
    return actions


def main():
    env = gym.make('Freeway-ram-v0')
    aspace = 3
    env.env.get_action_meanings()

    vmin,vmax = -10,10
    replay_size = 200
    batch_size = 50
    nsup = 51
    dz = (vmax - vmin) / (nsup - 1)
    support = torch.linspace(vmin, vmax, nsup)

    replay = deque(maxlen=replay_size) # Experience replay buffer
    lr = 0.0001
    gamma = 0.1 # Discount factor
    epochs = 2500
    eps = 0.20 # Starting epsilon for epsilon-greedy policy
    eps_min = 0.05 #E Ending epsilon
    priority_level = 5 # Prioritized replay
    update_freq = 25 # Frequency of target network updates

    #Initialize DQN parameter vector
    tot_params = 128 * 100 + 25 * 100 + aspace * 25 * 51
    theta = torch.randn(tot_params) / 10. # Random init
    theta.requires_grad=True
    theta_2 = theta.detach().clone()

    losses = []
    cum_rewards = [] # Stores each win as one
    state = preproc_state(env.reset())

    for i in range(epochs):
        pred = dist_dqn(state,theta,aspace=aspace)
        if i < replay_size or np.random.rand(1) < eps: # Epsilon greedy action selection
            action = np.random.randint(aspace)
        else:
            action = get_action(pred.unsqueeze(dim=0).detach(),support).item()
        state2, reward, done, _ = env.step(action)
        state2 = preproc_state(state2)
        if reward == 1: cum_rewards.append(1) 
        reward = 10 if reward == 1 else reward
        reward = -10 if done else reward
        reward = -1 if reward == 0 else reward
        exp = (state,action,reward,state2)
        replay.append(exp)
        
        if reward == 10: # Put more weight on success
            for _ in range(priority_level):
                replay.append(exp)
                
        shuffle(replay)
        state = state2

        if len(replay) == replay_size: # Begin training if buffer is full
            indx = np.random.randint(low=0,high=len(replay),size=batch_size)
            exps = [replay[j] for j in indx]
            state_batch = torch.stack([ex[0] for ex in exps],dim=1).squeeze()
            action_batch = torch.Tensor([ex[1] for ex in exps])
            reward_batch = torch.Tensor([ex[2] for ex in exps])
            state2_batch = torch.stack([ex[3] for ex in exps],dim=1).squeeze()
            pred_batch = dist_dqn(state_batch.detach(),theta,aspace=aspace)
            pred2_batch = dist_dqn(state2_batch.detach(),theta_2,aspace=aspace)
            target_dist = get_target_dist(pred2_batch,action_batch,reward_batch, \
                                        support, lim=(vmin,vmax),gamma=gamma)
            loss = lossfn(pred_batch,target_dist.detach())
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad(): # Gradient descent
                theta -= lr * theta.grad
            theta.requires_grad = True
            
        if i % update_freq == 0: # Syncronize target and actor networks
            theta_2 = theta.detach().clone()
            
        if i > 100 and eps > eps_min: # Decrement epsilon
            dec = 1./np.log2(i)
            dec /= 1e3
            eps -= dec
        
        if done:
            state = preproc_state(env.reset())
            done = False

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
