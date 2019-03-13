#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 10
# ----------------------------------------------------------------------

"""Experience reply.

Agent's experiences are stored in form of tuples:
    (state, action, reward, new_state, episode_finished, priority)
"""

import collections
import random

import numpy as np

# ----------------------------------------------------------------------
class Memory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []

        self.unique_samples = False

    def add(self, state, action, reward, new_state, is_done, priority=1.0):
        sample = [state, action, reward, new_state, is_done, priority]
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            self.samples.pop(0)

        return True

    def sample(self, sample_size):
        n = min(sample_size, len(self.samples))
        return zip(*random.sample(self.samples, n))

    def _is_unique(self, sample):
        pass

    def __len__(self):
        return len(self.samples)

    def calculate_probs(self, loss, indices):
        """Update samples probabilities.
        """

# ----------------------------------------------------------------------
class PrioritizedMemory(Memory):
    pass

# ----------------------------------------------------------------------
def _test():
    m = Memory(10)

    m.add(0, 1, 2, 3, 0)
    m.add(4, 5, 6, 7, 0)
    m.add(8, 9, 10, 11, 1)
    m.add(12, 13, 14, 15, 1)

    states, actions, rewards, new_states, dones, _ = m.sample(2)
    print(states, actions, rewards, new_states, dones)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    _test()