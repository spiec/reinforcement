#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019
# ----------------------------------------------------------------------

"""Wrapper for the Pong game implementation.
"""

from game import PongGame

# ----------------------------------------------------------------------
class PongEnvironment(object):

    def __init__(self, seed=999):
        super(PongEnvironment, self).__init__()
        self.game = PongGame(render=False)
    
    def info(self) -> (int, int):
        return self.game.state_size(), self.game.n_actions()

    def reset(self):
        return self.game.reset()

    def step(self, state, agent):
        # agent-environment interaction step
        action_idx, expected_reward = agent.action(state, allow_random=True)   # left/right/do nothing
        reward, new_state, is_done = self.game.player_action(action_idx)

        return reward, action_idx, new_state, is_done

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pass