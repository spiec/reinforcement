#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:        sebastian.piec@
# Last modified: 2019
# ----------------------------------------------------------------------

"""
"""

import copy
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

# ----------------------------------------------------------------------
class EvoAgent(object):

    def __init__(self, state_size, n_actions, architecture=None, use_cuda=True):
        super(EvoAgent, self).__init__()

        self.state_size = state_size
        self.n_actions = n_actions
        self.architecture = architecture
        
        self.fitness = 0
        self.max_episode_steps = 10000
        self.net = None

        self._device = torch.device("cuda:0" if use_cuda else "cpu")

    def reinitialize(self):
        self.net = self.architecture(self.state_size, self.n_actions)
        self.net.to(self._device)

    def action(self, state, allow_random=False):
        """
        Returns:
            action_idx (int), index of action with expected highest total reward
            reward (float), expected total reward when executing this action in the state
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self._device)
            actions_q = self._predict_one(state_t).data.cpu().numpy()
            action_idx = np.argmax(actions_q)
            reward = actions_q[action_idx]

        return action_idx, reward

    def _predict(self, examples_batch):
        """
        Returns:
            Q(s, a) vector(s)
        """
        return self.net(examples_batch)
        
    def _predict_one(self, state):
        if type(self.state_size) == int:
            return self._predict(state.reshape(1, self.state_size)).flatten()
        return self._predict(state.reshape(1, *self.state_size)).flatten()
    
    def evaluate(self, environment, max_reward=1.0):
        """Play full episode, calculate accumulated reward.
        """
        episode_reward = 0.0
        state = environment.reset()

        for step_idx in range(self.max_episode_steps):
            reward, action_idx, new_state, is_done = environment.step(state, self)
        
            state = new_state
            episode_reward += reward

            if is_done or episode_reward >= max_reward:
                break

        self.fitness = episode_reward
        return episode_reward

    def mutate(self, noise_std=0.01):
        agent = copy.deepcopy(self)
        new_net = copy.deepcopy(self.net)

        for p in new_net.parameters():
            noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32)).to(self._device)
            p.data += noise_std * noise_t

        agent.net = new_net
        agent.net.to(self._device)
        return agent

        # NES (in other module)

    def save(self, filename):
        print("Saving {}".format(filename))
        torch.save(self.net, filename)

    def load(self, filename):
        print("Loading {}".format(filename))
        self.net = torch.load(filename)
        self.net.to(self._device)
        self.net.eval()             # important when dropout and batchnorm are used

    def random_update(self, noise):
        """
        """

