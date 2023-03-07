import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0) # Returns log probs over the two actions
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c)) # Returns values between -1 and 1
        return actor, critic
    

def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) # Each worker has own environment and optimizer
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env,worker_model) #B plays one episode
        actor_loss,critic_loss, eplen = update_params(worker_opt,values,logprobs,rewards) # Use collected data to run one update step
        counter.value = counter.value + 1 # Globally shared counter


def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float() # Converts state to tensor
    values, logprobs, rewards = [], [], []
    done = False
    j=0
    while not done:
        j+=1
        policy, value = worker_model(state) # Computes state value and log probs over actions
        values.append(value)
        logits = policy.view(-1) # Flattens
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample() # Samples from action probability distribution
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, _ = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)
    return values, logprobs, rewards


def update_params(worker_opt,values,logprobs,rewards,clc=0.1,gamma=0.95):
        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) # Reverse and flatten
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)
        Returns = []
        ret_ = torch.Tensor([0])
        for r in range(rewards.shape[0]): # Compute return values
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        actor_loss = -1*logprobs * (Returns - values.detach()) # Prevent backprop through critic head
        critic_loss = torch.pow(values - Returns,2)
        loss = actor_loss.sum() + clc*critic_loss.sum() # sum both losses to get overall loss. scale critic loss to ensure critic learns slower
        loss.backward()
        worker_opt.step()
        return actor_loss, critic_loss, len(rewards)


if __name__ == '__main__':
    MasterNode = ActorCritic() # Shared AC model
    MasterNode.share_memory() # Allows params to be shared accross processes
    processes = []
    params = {
        'epochs':1000,
        'n_workers':7,
    }

    counter = mp.Value('i',0) # Shared global counter
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i,MasterNode,counter,params)) # Starts new process that runs a worker
        p.start() 
        processes.append(p)
    for p in processes: # Joins each process to wait for it to finish before returning to main function
        p.join()
    for p in processes:
        p.terminate()
        
    print(counter.value,processes[1].exitcode)

    # eval
    env = gym.make("CartPole-v1")
    env.reset()

    for i in range(100):
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        logits,value = MasterNode(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        state2, reward, done, info = env.step(action.detach().numpy())
        if done:
            print("Lost")
            env.reset()
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()