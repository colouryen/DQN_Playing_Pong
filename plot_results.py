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

from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
RS = 123


##### Reward Plot #####

R = np.array([])
L = np.array([])
hLayers = np.array([])
a_list = np.array([])
if op.isfile('Results.mat'):
    loadData = sio.loadmat('Results.mat')
    R = loadData['reward_list']
    R.shape = (-1, 1)
    L = loadData['loss_list']
    L.shape = (-1, 1)
    hLayers = loadData['hiddenLayers']
    hLayers.shape = (-1, 1)
    a_list = loadData['action_list']
    a_list.shape = (-1, 1)
    
#hLayers = np.array(hLayers)
#a_list = np.array(a_list)

t = np.arange(1, R.size + 1, 1)

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

##### Data Representation #####
# choose a color palette with seaborn.
Layer_embedded = TSNE(n_components=2).fit_transform(hLayers)
num_classes = len(np.unique(a_list))
palette = np.array(sns.color_palette("hls", num_classes))

# create a scatter plot.
f = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sc = ax.scatter(Layer_embedded[:,0], Layer_embedded[:,1], lw=0, s=40, c=palette[a_list.astype(np.int)])
plt.xlim(-25, 25)
plt.ylim(-25, 25)
ax.axis('off')
ax.axis('tight')

# add the labels for each digit corresponding to the label
txts = []

for i in range(num_classes):

    # Position of each label at median of data points.

    xtext, ytext = np.median(Layer_embedded[a_list == i, :], axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
    txts.append(txt)


'''
plt.scatter(Layer_embedded[len(hiddenLayers):,0], Layer_embedded[len(hiddenLayers):,1], marker='.')
sp = plt.scatter(Layer_embedded[len(hiddenLayers):,0], Layer_embedded[len(hiddenLayers):,1], c=y1)
#plt.scatter(Layer_embedded[:, 0], Layer_embedded[:, 1], marker="x", cmap=plt.get_cmap('Spectral'))
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding hidden layer')
plt.savefig('t-SNE_hidden_layer.png')
'''