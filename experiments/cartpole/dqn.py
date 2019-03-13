#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:        sebastian.piec@
# Last modified: 2019
# ----------------------------------------------------------------------

"""Deep Q-Network agent playing cart pole game.
"""

import sys
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from rltools.dqn.agent import DqnAgent
from rltools.dqn.exploit import EpsilonGreedy

# ----------------------------------------------------------------------
class CartAgent(DqnAgent):

    def __init__(self, state_size, n_actions, filename=None):
        super(CartAgent, self).__init__(state_size, n_actions, memory_size=5000,
                                        architecture=CartNet)
        self.batch_size = 32 
        self.discount = 0.95
        self.learning_rate = 0.001
        self.target_net_update = 1  #50
        self.loss_function = F.mse_loss
        self.action_selector = EpsilonGreedy(1.0, 0.01, epsilon=get_epsilon)

        if filename:
            self.load(filename)
        self._reinitialize()

    def episode_finished(self, episode_idx, n_episodes):
        self.action_selector.adjust(progress=float(episode_idx) / n_episodes)
        super(CartAgent, self).episode_finished(episode_idx, n_episodes)

# ----------------------------------------------------------------------
def get_epsilon(progress, max_epsilon, min_epsilon):
    return np.exp(-2 * progress)

# ----------------------------------------------------------------------
class CartNet(nn.Module):

    def __init__(self, state_size, n_actions):
        super(CartNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, n_actions),
        )
        print(self)

    def forward(self, x):
        return self.fc(x)

# ----------------------------------------------------------------------
def training_loop(agent, environment, config, callbacks):
    max_steps = int(10e+5)          # max steps per episode
    interval = 10                  # for logs and callbacks
    rewards = []
    eta = 0

    for episode_idx in range(config.n_episodes):
        print("\n", "=" * 20, " NEW EPISODE ", "=" * 20, "\n")
            
        state = environment.reset()
        episode_reward = 0.0
        
        for step_idx in range(max_steps):
            reward, action_idx, new_state, is_done = environment.step(state, agent)
            agent.memorize(state, action_idx, reward, new_state, is_done)
            #agent.training_step()

            state = new_state
            episode_reward += reward

            if is_done:
                print("DONE! Episode reward:", reward)
                break
        
        rewards.append(episode_reward)
        agent.training_step()
        
        if (episode_idx + 1) % interval == 0:
            mean_reward = np.mean(rewards[-interval:])
            sys.stderr.write("Episode {0}/{1} ({2:.2f}% ETA: {3:.2f}), reward: {4}, steps: {5}, eps: {6:.4f}, mean reward: {7:.2f}\n".format(
                             episode_idx, config.n_episodes, float(episode_idx) / config.n_episodes * 100, eta,
                             episode_reward, step_idx, agent.action_selector.epsilon,
                             mean_reward))
                            
            for callback in callbacks:
                callback((mean_reward, agent.action_selector.epsilon, episode_idx))
        agent.episode_finished(episode_idx, config.n_episodes)

    return agent