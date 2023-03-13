import gym
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skimage.transform import resize
import torch
from torch import nn
from einops import rearrange
from collections import deque
from matplotlib import pyplot as plt


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=(4,4))
        self.conv2 = nn.Conv2d(10,16,kernel_size=(4,4))
        self.conv3 = nn.Conv2d(16,24,kernel_size=(4,4))
        self.conv4 = nn.Conv2d(24,32,kernel_size=(4,4))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv5 = nn.Conv2d(32,64,kernel_size=(4,4))
        self.lin1 = nn.Linear(256,128)
        self.out = nn.Linear(128,10)
    def forward(self,x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        x = nn.functional.log_softmax(x,dim=1)
        return x


class MultiHeadRelationalModule(torch.nn.Module):
    def __init__(self):
        super(MultiHeadRelationalModule, self).__init__()
        self.conv1_ch = 16 
        self.conv2_ch = 20
        self.conv3_ch = 24
        self.conv4_ch = 30
        self.H = 28
        self.W = 28
        self.node_size = 64
        self.lin_hid = 100
        self.out_dim = 5
        self.ch_in = 3
        self.sp_coord_dim = 2
        self.N = int(7**2)
        self.n_heads = 3
        
        self.conv1 = nn.Conv2d(self.ch_in,self.conv1_ch,kernel_size=(1,1),padding=0) # 1x1 conv to preserve spational organization of object on grid
        self.conv2 = nn.Conv2d(self.conv1_ch,self.conv2_ch,kernel_size=(1,1),padding=0)
        self.proj_shape = (self.conv2_ch+self.sp_coord_dim,self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.k_lin = nn.Linear(self.node_size,self.N) # Layers of additive attention
        self.q_lin = nn.Linear(self.node_size,self.N)
        self.a_lin = nn.Linear(self.N,self.N)
        
        self.node_shape = (self.n_heads, self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        
        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N,self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)
    
    def forward(self,x):
        N, Cin, H, W = x.shape
        x = self.conv1(x) 
        x = torch.relu(x)
        x = self.conv2(x) 
        x = torch.relu(x) 
        with torch.no_grad(): 
            self.conv_map = x.clone() # Save for visualization
        _,_,cH,cW = x.shape
        xcoords = torch.arange(cW).repeat(cH,1).float() / cW
        ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
        spatial_coords = torch.stack([xcoords,ycoords],dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N,1,1,1)
        x = torch.cat([x,spatial_coords],dim=1)
        x = x.permute(0,2,3,1)
        x = x.flatten(1,2)
        
        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K) 
        
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q) 
        
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V) 
        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K)) # additive attention
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A,dim=3) 
        with torch.no_grad():
            self.att_map = A.clone() # Save attention weights for visualization
        E = torch.einsum('bhfc,bhcd->bhfd',A,V) # Batch matrix multiplies the attention weight with the node matrix to update the node matrix
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y)
        return y
    

def prepare_state(x): # Normalizes state tensor and converts to pytorch tensor
    ns = torch.from_numpy(x).float().permute(2,0,1).unsqueeze(dim=0)#
    maxv = ns.flatten().max()
    ns = ns / maxv
    return ns


def get_minibatch(replay,size): # Samples minibatch from experience replay
    batch_ids = np.random.randint(0,len(replay),size)
    batch = [replay[x] for x in batch_ids] #list of tuples
    state_batch = torch.cat([s for (s,a,r,s2,d) in batch],)
    action_batch = torch.Tensor([a for (s,a,r,s2,d) in batch]).long()
    reward_batch = torch.Tensor([r for (s,a,r,s2,d) in batch])
    state2_batch = torch.cat([s2 for (s,a,r,s2,d) in batch],dim=0)
    done_batch = torch.Tensor([d for (s,a,r,s2,d) in batch])
    return state_batch,action_batch,reward_batch,state2_batch, done_batch


def get_qtarget_ddqn(qvals,r,df,done): # Calculates the target Q value
    targets = r + (1-done) * df * qvals
    return targets


def lossfn(pred,targets,actions):
    loss = torch.mean(torch.pow(\
                                targets.detach() -\
                                pred.gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()\
                                ,2),dim=0)
    return loss
  

def update_replay(replay,exp,replay_size): # Adds to experience replay
    r = exp[2]
    N = 1
    if r > 0: # add 50 copies for positive rewards
        N = 50
    for _ in range(N):
        replay.append(exp)
    return replay

action_map = {
    0:0, 
    1:1,
    2:2,
    3:3,
    4:5,
}


if __name__ == '__main__':
    env = ImgObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0'))
    state = prepare_state(env.reset()) 
    GWagent = MultiHeadRelationalModule() # Main
    Tnet = MultiHeadRelationalModule() # Target
    maxsteps = 400 # Game timeout
    env.max_steps = maxsteps
    env.env.max_steps = maxsteps

    epochs = 50000
    replay_size = 9000
    batch_size = 50
    lr = 0.0005
    gamma = 0.99
    replay = deque(maxlen=replay_size) # Experience replay buffer
    opt = torch.optim.Adam(params=GWagent.parameters(),lr=lr)
    eps = 0.5
    update_freq = 100
    for i in range(epochs):
        pred = GWagent(state)
        action = int(torch.argmax(pred).detach().numpy())
        if np.random.rand() < eps: # Epsilon greedy
            action = int(torch.randint(0,5,size=(1,)).squeeze())
        action_d = action_map[action]
        state2, reward, done, info = env.step(action_d)
        reward = -0.01 if reward == 0 else reward # Rescales rewards on non-terminal states
        state2 = prepare_state(state2)
        exp = (state,action,reward,state2,done)
        
        replay = update_replay(replay,exp,replay_size)
        if done:
            state = prepare_state(env.reset())
        else:
            state = state2
        if len(replay) > batch_size:
            
            opt.zero_grad()
            
            state_batch,action_batch,reward_batch,state2_batch,done_batch = get_minibatch(replay,batch_size)
            
            q_pred = GWagent(state_batch).cpu()
            astar = torch.argmax(q_pred,dim=1)
            qs = Tnet(state2_batch).gather(dim=1,index=astar.unsqueeze(dim=1)).squeeze()
            
            targets = get_qtarget_ddqn(qs.detach(),reward_batch.detach(),gamma,done_batch)
            
            loss = lossfn(q_pred,targets.detach(),action_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(GWagent.parameters(), max_norm=1.0) # Prevents overly large gradients
            opt.step()
        if i % update_freq == 0: # Updates target network
            Tnet.load_state_dict(GWagent.state_dict())

    state_ = env.reset()
    state = prepare_state(state_)
    GWagent(state)
    plt.imshow(env.render('rgb_array'))
    plt.imshow(state[0].permute(1,2,0).detach().numpy())
    head, node = 2, 26
    plt.imshow(GWagent.att_map[0][head][node].view(7,7))
