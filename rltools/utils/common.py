#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:        sebastian.piec@
# Last modified: 2018, August 25
# ----------------------------------------------------------------------

"""
"""

import time

# ----------------------------------------------------------------------
class GymWrapper(object):

    def __init__(self, gym_environment, seed=42, render=False):
        self._env = gym_environment
        self._seed = seed
        self._render = render
        self._env.seed(seed)

    def reset(self):
        state = self._env.reset()
        return state
        
    def step(self, state, agent):
        action, _ = agent.action(state, allow_random=True)
        next_state, reward, done, _ = self._env.step(action)
        if self._render:
            self._env.render()
        
        return reward, action, next_state, done

    def info(self):
        state_size = self._env.observation_space.shape[0]
        n_actions = self._env.action_space.n
        return state_size, n_actions
    
    def close(self):
        self._env.close()

# ----------------------------------------------------------------------
def play_episode(environment, agent=None, render=False):
    state = environment.reset()
    
    cnt = 0
    is_done = False 
    while not is_done:

        if agent:
            action, expected_reward = agent.best_action(state, environment)
        else:
            action = 0
            expected_reward = 0

        next_state, reward, is_done, info = environment._env.step(action)

        if render:
            environment._env.render()

        state = next_state
        cnt += 1
        
    print("Episode finished! # of steps: {}".format(cnt))