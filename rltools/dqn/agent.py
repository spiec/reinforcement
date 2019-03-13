#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 10
# ----------------------------------------------------------------------

"""Deep Q-network (DQN) approximating Q(s, a) function of MDP.

Implements a few tricks improving convergence speed and stability:
- experience replay (dec)
- prioritized experience replay, more surprising transitions
      are sampled more frequently
- target network, separate network is used for target Q(s', a') generation
- double Q-learning
- dueling network, decomposition of Q(s, a) = V(s) + A(a)
"""

import copy
import sys

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..agent import Agent
from .memory import Memory

# ----------------------------------------------------------------------
class DqnAgent(Agent):

    # ----------------------------------------------------------------------
    def __init__(self, state_size, n_actions, memory_size=10000, use_cuda=True,
                 architecture=None):
        super(DqnAgent, self).__init__()
        
        self.state_size = state_size
        self.n_actions = n_actions

        self.loss_function = F.smooth_l1_loss       # aka Huber loss
        
        # DQN hyperparams
        self.batch_size = 32 
        self.learning_rate = 0.01
        self.target_net_update = 100
        
        self.memory = Memory(memory_size)

        if use_cuda:
            if not torch.cuda.is_available():
                sys.stderr.write("WARNING! No CUDA device available. CPU will be used")
                use_cuda = False
        print("Use CUDA:", use_cuda)
        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        self.architecture = architecture
        #self._reinitialize()

    def _reinitialize(self):
        self.policy_net = self.architecture(self.state_size, self.n_actions).to(self._device)
        self.target_net = self.architecture(self.state_size, self.n_actions).to(self._device)
        
        self.optimizer = torch.optim.RMSprop(params=self.policy_net.parameters(),
                                             lr=self.learning_rate)
        #self.optimizer = torch.optim.SGD(params=self.policy_net.parameters(),
        #self.optimizer = torch.optim.Adam(params=self.policy_net.parameters(),

        self._training_step_cnt = 0

    def action(self, state, allow_random=False):
        """
        Returns:
            action_idx (int), index of action to be taken
            reward (float), expected total reward when executing this action in the state
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self._device)

            actions_q = self._predict_one(state_t).data.cpu().numpy()

            if not allow_random:
                action_idx = np.argmax(actions_q)                   # highest return action
            else:
                action_idx = self.action_selector(actions_q)        # e.g. e-greedy

            reward = actions_q[action_idx]

        return action_idx, reward

    def memorize(self, state, action_idx, reward, new_state, is_done):
        self.memory.add(state, action_idx, reward, new_state, is_done)

    def training_step(self):
        """Adjust agent's policy as soon as new bit of experience is available.
        """
        if len(self.memory) < self.batch_size:
            return

        states_v, actions_v, rewards_v, next_states_v, dones_v = self._prepare_batch()

        # batch of Q(s, a) values (row: experience idx, col: action idx)
        Q_values = self.policy_net(states_v).gather(1, actions_v.unsqueeze(1)).squeeze()
       
        # temporar "training target" for the "policy network"
        next_Q_values = self._next_state_Q(next_states_v, dones_v)

        # Bellman update
        target_Q_values = rewards_v + self.discount * next_Q_values

        loss = self.loss_function(Q_values, target_Q_values)
        #self._update_priorities(loss)               # of samples stored in the reply memory

        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping (REF: )
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        
        self._training_step_cnt += 1
        # target network flipping (REF: )
        if (self._training_step_cnt % self.target_net_update) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def _next_state_Q(self, next_states_v, dones_v):
        """Try to estimate Q(s,a) of the next states which is agent's training target.
        """
        #return self._double_q(next_states_v, dones_v)

        next_Q_values = self.target_net(next_states_v).max(1)[0]
        next_Q_values[dones_v] = 0.0        # Q for the terminal state is equal to actual reward
        next_Q_values = next_Q_values.detach()
        
        return next_Q_values

    def _double_q(self, next_states_v, dones_v):
        """Double Q-learning
        """
        # use policy network for action selection
        actions = self.policy_net(next_states_v).max(1)[1]              # ???? TODO
        
        # estimate Q of the next state using target network
        Q_next_values = self.target_net(next_states_v).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        Q_next_values[dones_v] = 0.0
        Q_next_values = Q_next_values.detach()

        return Q_next_values

    def _prepare_batch(self):
        states, actions, rewards, new_states, dones, self._sample_indices = self.memory.sample(self.batch_size)

        states_v = torch.Tensor(states).to(self._device)
        actions_v = torch.LongTensor(actions).to(self._device)
        rewards_v = torch.Tensor(rewards).to(self._device)
        new_states_v = torch.Tensor(new_states).to(self._device)
        dones_v = torch.ByteTensor(dones).to(self._device)

        return states_v, actions_v, rewards_v, new_states_v, dones_v

    def _update_priorities(self, loss_v):
        pass

    def episode_finished(self, episode_idx, n_episodes):
        super(DqnAgent, self).episode_finished(episode_idx, n_episodes)
        #for g in self.optimizer.param_groups:
        #    g["lr"] = lr

    def _predict(self, examples_batch):
        return self.policy_net(examples_batch)

    def _predict_one(self, state):
        if type(self.state_size) == int:
            return self._predict(state.reshape(1, self.state_size)).flatten()
        
        return self._predict(state.reshape(1, *self.state_size)).flatten()
    
    def save(self, filename):
        print("Saving agent {}".format(filename))
        torch.save(self.policy_net, filename)

    def load(self, filename):
        print("Loading agent {}".format(filename))

        self.policy_net = torch.load(filename)
        self.policy_net.to(self._device)
        self.policy_net.eval()

        self.target_net = TargetNet(self.policy_net).to(self._device)
        #self.target_net.eval()

# ----------------------------------------------------------------------
class TargetNet(object):

    def __init__(self, base_net):
        self._base_net = base_net
        self._target_net = copy.deepcopy(base_net)

    def __call__(self, x):
        return self._target_net(x)

    def to(self, device):
        self._target_net.to(device)
        return self

    def sync(self):
        self._target_net.load_state_dict(self._base_net.state_dict())