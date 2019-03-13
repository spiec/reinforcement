#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 10
# ----------------------------------------------------------------------

"""
"""

import sys
import numpy as np

# ----------------------------------------------------------------------
def basic_training(agent, environment, n_episodes, callbacks=[],
                   max_steps=100000, log_interval=100):
    rewards = []

    for episode_idx in range(n_episodes):
        state = environment.reset()
        print("\n", "=" * 20, " NEW EPISODE ", "=" * 20, "\n")
            
        episode_reward = 0.0
        
        for step_idx in range(max_steps):
            reward, action_idx, new_state, is_done = environment.step(state, agent)

            # if DQN agent 
            agent.memorize(state, action_idx, reward, new_state, is_done)

            state = new_state
            episode_reward += reward

            if is_done:
                break

        rewards.append(episode_reward)
        agent.training_step()
        
        if (episode_idx + 1) % log_interval == 0:
            mean_reward = np.mean(rewards[-log_interval:])
            sys.stderr.write("Episode {0}/{1}, reward: {2}, steps: {3}, eps: {4:.4f}, mean reward: {5:.2f}\n".format(
                             episode_idx, n_episodes, episode_reward, step_idx, agent.action_selector.epsilon,
                             mean_reward))
                            
            for callback in callbacks:
                callback((mean_reward, agent.epsilon, episode_idx))

        agent.episode_finished(episode_idx, n_episodes)

    return agent