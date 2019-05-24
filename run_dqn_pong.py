from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

import scipy.io as sio
import os.path as op
import matplotlib.pyplot as plt

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1000000
batch_size = 32
gamma = 0.99
    
replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if USE_CUDA:
    model = model.cuda()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

loss_list = []
reward_list = []

state = env.reset()

frame_list = random.sample(range(1, num_frames), 1000)
frame_list.sort()
action_list = []
state_list = []
hiddenLayers = []
rand_frame_count = 0

for frame_idx in range(1, num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    
    if frame_list[rand_frame_count] == frame_idx:
        state_list.append(state.squeeze(0))
        
        hiddenTensor = model.get_hidden_layer(state)
        temp = hiddenTensor.data.cpu().numpy()
        hiddenLayers.append(temp[0])
        #hiddenLayers.append(hiddenTensor.data.cpu().numpy())
        #hiddenLayers = np.concatenate((hiddenLayers, hiddenTensor.data.cpu().numpy()), axis=0)
        
        action_list.append(action)
        #env.env.ale.saveScreenPNG('test_image.png')
        
        if rand_frame_count < 999:
            rand_frame_count += 1
    
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().numpy())
        
    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))
    
        loss_list.append(np.mean(losses)) 
        reward_list.append(np.mean(all_rewards[-10:]))

   
sio.savemat('Results.mat', {'reward_list':reward_list, 'loss_list':loss_list, 'hiddenLayers':hiddenLayers, 'action_list':action_list, 'state_list':state_list})  

#hiddenLayers = np.array(hiddenLayers)
#action_list = np.array(action_list)

#Layer_embedded = TSNE(n_components=2).fit_transform(hiddenLayers)

'''
plt.scatter(Layer_embedded[len(hiddenLayers):,0], Layer_embedded[len(hiddenLayers):,1], marker='.')
sp = plt.scatter(Layer_embedded[len(hiddenLayers):,0], Layer_embedded[len(hiddenLayers):,1], c=y1)
#plt.scatter(Layer_embedded[:, 0], Layer_embedded[:, 1], marker="x", cmap=plt.get_cmap('Spectral'))
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding hidden layer')
plt.savefig('t-SNE_hidden_layer.png')
'''

#fashion_scatter(Layer_embedded, action_list)
