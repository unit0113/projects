import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque
from random import shuffle


BATCH_SIZE = 150
BETA = 0.2
LAMBDA = 0.1
ETA = 1.0
GAMMA = 0.2
MAX_EPISODE_LEN = 100
MIN_PROGRESS = 15
ACTION_REPEATS = 6
FPS = 3


def downscale_obs(obs, new_size=(42,42), to_gray=True):
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    else:
        return resize(obs, new_size, anti_aliasing=True)
    

# Downscales, converts to grayscale, adds batch dimension
def prepare_state(state):
    return torch.from_numpy(downscale_obs(state, to_gray=True)).float().unsqueeze(dim=0)


# Adds latest frame to queue
def prepare_multi_state(state1, state2):
    state1 = state1.clone()
    tmp = torch.from_numpy(downscale_obs(state2, to_gray=True)).float()
    state1[0][0] = state1[0][1]
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp
    return state1


# Creates initial state of 3x first frame, adds batch dimension
def prepare_initial_state(state,N=3):
    state_ = torch.from_numpy(downscale_obs(state, to_gray=True)).float()
    tmp = state_.repeat((N,1,1))
    return tmp.unsqueeze(dim=0)


def policy(qvalues, eps=None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0,high=7,size=(1,))
        else:
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues)), num_samples=1) # sample from softmax
    

class ExperienceReplay:
    def __init__(self, N=500, batch_size=100):
        self.N = N # Max size
        self.batch_size = batch_size # number of samples to return when training
        self.memory = [] 
        self.counter = 0
        
    def add_memory(self, state1, action, reward, state2):
        self.counter +=1 
        if self.counter % 500 == 0: # Promotes more randomized sample
            self.shuffle_memory()
            
        if len(self.memory) < self.N: # Add if not full, else replace random with new
            self.memory.append( (state1, action, reward, state2) )
        else:
            rand_index = np.random.randint(0,self.N-1)
            self.memory[rand_index] = (state1, action, reward, state2)
    
    def shuffle_memory(self):
        shuffle(self.memory)
        
    def get_batch(self): # Samples mini-batch
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        if len(self.memory) < 1:
            print("Error: No data in memory.")
            return None
        #G
        ind = np.random.choice(np.arange(len(self.memory)),batch_size,replace=False)
        batch = [self.memory[i] for i in ind] # batch is a list of tuples
        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch],dim=0)
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch],dim=0)
        return state1_batch, action_batch, reward_batch, state2_batch
    

# Encode network
class Phi(nn.Module):
    def __init__(self):
        super(Phi, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

    def forward(self,x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y)) # size [1, 32, 3, 3] batch, channels, 3 x 3
        y = y.flatten(start_dim=1) # size N, 288
        return y


# Inverse model
class Gnet(nn.Module):
    def __init__(self):
        super(Gnet, self).__init__()
        self.linear1 = nn.Linear(576, 256)
        self.linear2 = nn.Linear(256, 12)

    def forward(self, state1,state2):
        x = torch.cat( (state1, state2) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y, dim=1)
        return y


# Forward model
class Fnet(nn.Module):
    def __init__(self):
        super(Fnet, self).__init__()
        self.linear1 = nn.Linear(300,256)
        self.linear2 = nn.Linear(256,288)

    def forward(self,state,action):
        action_ = torch.zeros(action.shape[0],12) # one-hot encoding
        indices = torch.stack( (torch.arange(action.shape[0]), action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat( (state,action_) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y
    

class Qnetwork(nn.Module):
    def __init__(self):
        super(Qnetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.linear1 = nn.Linear(288,100)
        self.linear2 = nn.Linear(100,12)
        
    def forward(self,x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim=2)
        y = y.view(y.shape[0], -1, 32)
        y = y.flatten(start_dim=1)
        y = F.elu(self.linear1(y))
        y = self.linear2(y) #size N, 12
        return y


def loss_fn(q_loss, inverse_loss, forward_loss):
    loss_ = (1 - BETA) * inverse_loss
    loss_ += BETA * forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + LAMBDA * q_loss
    return loss

def reset_env(env):
    """
    Reset the environment and return a new initial state
    """
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    return state1


def ICM(state1, action, state2, encoder, forward_model, forward_loss, inverse_model, inverse_loss, forward_scale=1., inverse_scale=1e4):
    state1_hat = encoder(state1)
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(), action.detach())
    forward_pred_err = forward_scale * forward_loss(state2_hat_pred, state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat, state2_hat) # Returns softmax over actions
    inverse_pred_err = inverse_scale * inverse_loss(pred_action, \
                                        action.detach().flatten()).unsqueeze(dim=1)
    return forward_pred_err, inverse_pred_err


def minibatch_train(replay, Qmodel, qloss, encoder, forward_model, forward_loss, inverse_model, inverse_loss, use_extrinsic=True):
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    action_batch = action_batch.view(action_batch.shape[0],1)
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    
    forward_pred_err, inverse_pred_err = ICM(state1_batch, action_batch, state2_batch, encoder, forward_model, forward_loss, inverse_model, inverse_loss) # Run ICM
    i_reward = (1. / ETA) * forward_pred_err
    reward = i_reward.detach()

    if use_extrinsic:
        reward += reward_batch 
    qvals = Qmodel(state2_batch) # Action values for next state
    reward += GAMMA * torch.max(qvals)
    reward_pred = Qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack( (torch.arange(action_batch.shape[0]), action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), F.normalize(reward_target.detach()))

    return forward_pred_err, inverse_pred_err, q_loss


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    replay = ExperienceReplay(N=1000, batch_size=BATCH_SIZE)
    Qmodel = Qnetwork()
    encoder = Phi()
    forward_model = Fnet()
    inverse_model = Gnet()
    forward_loss = nn.MSELoss(reduction='none')
    inverse_loss = nn.CrossEntropyLoss(reduction='none')
    qloss = nn.MSELoss()
    all_model_params = list(Qmodel.parameters()) + list(encoder.parameters()) #A
    all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
    opt = optim.Adam(lr=0.001, params=all_model_params)

    epochs = 5000
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    eps=0.15
    losses = []
    episode_length = 0
    switch_to_eps_greedy = 1000
    state_deque = deque(maxlen=FPS)
    e_reward = 0.
    last_x_pos = env.env.env._x_position # Keep track of in order to reset if no forward progress
    ep_lengths = []
    use_extrinsic = True

    for i in range(epochs):
        opt.zero_grad()
        episode_length += 1
        q_val_pred = Qmodel(state1) # Get action-value predictions
        if i > switch_to_eps_greedy: # Switch to eps greddy after 1000 epochs
            action = int(policy(q_val_pred, eps))
        else:
            action = int(policy(q_val_pred))
        for _ in range(ACTION_REPEATS): # Repeat X times to speed up learning
            state2, e_reward_, done, info = env.step(action)
            last_x_pos = info['x_pos']
            if done:
                state1 = reset_env(env)
                break
            e_reward += e_reward_
            state_deque.append(prepare_state(state2))

        state2 = torch.stack(list(state_deque),dim=1) # Convert to tensor
        replay.add_memory(state1, action, e_reward, state2)
        e_reward = 0

        if episode_length > MAX_EPISODE_LEN: # Restart if timeout
            if (info['x_pos'] - last_x_pos) < MIN_PROGRESS:
                done = True
            else:
                last_x_pos = info['x_pos']
        if done:
            ep_lengths.append(info['x_pos'])
            state1 = reset_env(env)
            last_x_pos = env.env.env._x_position
            episode_length = 0
        else:
            state1 = state2
        if len(replay.memory) < BATCH_SIZE:
            continue

        forward_pred_err, inverse_pred_err, q_loss = minibatch_train(replay, Qmodel, qloss, encoder, forward_model, forward_loss, inverse_model, inverse_loss, use_extrinsic) # Sample replay buffer
        loss = loss_fn(q_loss, forward_pred_err, inverse_pred_err) # Compute overall loss
        loss_list = (q_loss.mean(), forward_pred_err.flatten().mean(),\
        inverse_pred_err.flatten().mean())
        losses.append(loss_list)
        loss.backward()
        opt.step()

    done = True
    state_deque = deque(maxlen=FPS)
    for _ in range(5000):
        if done:
            env.reset()
            state1 = prepare_initial_state(env.render('rgb_array'))
        q_val_pred = Qmodel(state1)
        action = int(policy(q_val_pred,eps))
        state2, _, done, _ = env.step(action)
        state2 = prepare_multi_state(state1,state2)
        state1=state2
        env.render()

    losses_ = losses.detach.numpy()
    plt.figure(figsize=(8,6))
    plt.plot(np.log(losses_[:,0]),label='Q loss')
    plt.plot(np.log(losses_[:,1]),label='Forward loss')
    plt.plot(np.log(losses_[:,2]),label='Inverse loss')
    plt.legend()
    plt.show()





if __name__ == '__main__':
    main()