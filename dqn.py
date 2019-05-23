from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        '''
        self.fc_temp = nn.Sequential(
            nn.Linear(self.feature_size(), 512)
        )
        '''
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            ######## YOUR CODE HERE! ########
            # TODO: Given state, you should write code to get the Q value and chosen action
            # Complete the R.H.S. of the following 2 lines and uncomment them
            q_value = self.forward(state)
            action = q_value.max(1)[1].view(1, 1)
            #action = action.item()
            ######## YOUR CODE HERE! ########
        else:
            action = random.randrange(self.env.action_space.n)
        return action
        
def compute_td_loss(model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    
    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss
    '''
    non_final_mask = Variable(torch.ByteTensor(np.uint8(tuple(map(lambda s: s is not None, next_state)))))
    #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)), device=self.device, dtype=torch.uint8)
    try: #sometimes all next states are false
        non_final_next_states = Variable(torch.FloatTensor(np.float32([s for s in next_state if s is not None])))
        #non_final_next_states = torch.tensor([s for s in next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
        empty_next_state_values = False
    except:
        non_final_next_states = None
        empty_next_state_values = True
    '''
    current_q_value = model(state).gather(1, action.view(-1, 1))
    #max_next_action = model(next_state).max(dim=1)[1].view(-1, 1)
    expected_q_value = reward + (gamma * model(next_state).gather(1, action.view(-1, 1))) * (1 - done)
    
    '''
    #target
    with torch.no_grad():
        max_next_q_value = Variable(torch.zeros(batch_size, dtype=torch.float)).unsqueeze(dim=1)
        #max_next_q_value = torch.zeros(batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
        if not empty_next_state_values:
            max_next_action = model(non_final_next_states).max(dim=1)[1].view(-1, 1)
            max_next_q_value[non_final_mask] = model(non_final_next_states).gather(1, max_next_action)
        expected_q_values = reward + (gamma * max_next_q_value)
    '''
    err = (expected_q_value - current_q_value)
    
    loss = torch.mean(err**2)
    ######## YOUR CODE HERE! ########
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        # Hint: you may use the python library "random".
        # If you are not familiar with the "deque" python library, please google it.
        ######## YOUR CODE HERE! ########
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
