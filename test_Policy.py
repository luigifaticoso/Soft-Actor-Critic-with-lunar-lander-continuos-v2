import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        #log is clamped to be iin a san region. 
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(device))
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        #Then to get the action we use the reparameterization trick
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        #we sample a noise from a Standard Normal distribution
        normal = Normal(0, 1)
        #multiply it with our standard devation
        z      = normal.sample().to(device)
        #add it to the mean and make it activated with a tanh to give our function
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()
        return action[0]


policy_net = torch.load('modello500ep')

# def display_frames_as_gif(frames):
#     """
#     Displays a list of frames as a gif, with controls
#     """
#     #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
#     patch = plt.imshow(frames[0])
#     plt.axis('off')

#     def animate(i):
#         patch.set_data(frames[i])

#     anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
#     display(anim)

env = gym.make("LunarLanderContinuous-v2")

# Run a demo of the environment
tot_reward = 0
frames = []
top_reward = 0
for i in range(50):
    cum_reward = 0
    state = env.reset()
    for t in range(500):
        # Render into buffer. 
        env.render(mode = 'rgb_array')
        action = policy_net.get_action(state)
        state, reward, done, info = env.step(action.detach().numpy())
        cum_reward += reward
        if done:
            break
    print("reward for this episode {} is : {}".format(i,cum_reward))
    tot_reward += cum_reward
    top_reward = max(cum_reward,top_reward)
print("average per episode: {}".format(cum_reward))
print("top reward {}".format(top_reward))
env.close()
# display_frames_as_gif(frames)

# for t in range(50000):
#     # Render into buffer. 
#     env.render(mode = 'rgb_array')
#     action = policy_net.get_action(state)
#     state, reward, done, info = env.step(action.detach().numpy())
#     cum_reward += reward
#     if done:
#         break
# print("average per episode: {}".format(cum_reward))
# env.close()