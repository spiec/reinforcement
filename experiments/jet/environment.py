#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 11
# ----------------------------------------------------------------------

"""Wrapper for PyGame Jet implementation.
"""

from pygame_jet import JetGame

# ----------------------------------------------------------------------
class JetEnvironment(object):

    def __init__(self, render=True, seed=999):
        super(JetEnvironment, self).__init__()
        self.game = JetGame(render=render)
        
    def info(self) -> (int, int):
        return self.game.state_size(), self.game.n_actions()

    def reset(self):
        return self.game.reset()

    def step(self, state, agent):
        # agent-environment interaction step.
        action_idx, expected_reward = agent.action(state, allow_random=True)
        reward, new_state, is_done = self.game.player_action(action_idx)
        return reward, action_idx, new_state, is_done

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pass