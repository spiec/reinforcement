#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 10
# ----------------------------------------------------------------------

"""Base class for agents.
"""

import numpy as np

# ----------------------------------------------------------------------
class Agent(object):
    
    def __init__(self):
        self.discount = 0.9
        self.steps = 0

        self.action_selector = None

    def action(self, state, allow_random=False):
        raise NotImplementedError

    def training_step(self):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError

    def episode_finished(self, episode_idx, n_episodes):
        """All actions agent needs to perform when episode ends.
        """

# ----------------------------------------------------------------------
class RandomAgent(Agent):
    """Does not learn anything, moves randomly forever.
    """
    
    def __init__(self, state_size, n_actions):
        super(RandomAgent, self).__init__()

    def best_action(self, state, environment):
        return self.random_action(state, environment)

    def random_action(self, state, environment):
        """
        Returns:
            action, e.g. action's index (int) for discrete environments
            expected reward (float)
        """
        state_size, n_actions = environment.info()
        return np.random.choice(range(n_actions)), 0.0