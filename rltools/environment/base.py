#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 10
# ----------------------------------------------------------------------

"""Base class for RL environments.
"""

# ----------------------------------------------------------------------
class Environment(object):
    
    def __init__(self):
        super(Environment, self).__init__()

    def step(self, state, agent):
        raise NotImplementedError

    def reset(self):
        """Refresh the environment and return initial state at the beginning
        of new episode.
        """
        raise NotImplementedError

    def info(self):
        raise NotImplementedError