#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:47:54 2019

@author: ccyen
"""

import itertools
import numpy as np
import scipy.io as sio
import os.path as op
import matplotlib.pyplot as plt

##### Reward Plot #####

R = np.array([])
L = np.array([])
if op.isfile('Results.mat'):
    loadData = sio.loadmat('Results.mat')
    R = loadData['all_rewards']
    R.shape = (-1, 1)
    L = loadData['losses']
    L.shape = (-1, 1)

t = np.arange(1, R.szie + 1, 1)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('episodes')
ax1.set_ylabel('Reward', color=color)
ax1.plot(t, R, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
ax2.plot(t, L, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig('reward_loss_plot.png', bbox_inches='tight', dpi=300)