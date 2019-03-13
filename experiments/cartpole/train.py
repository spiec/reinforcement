#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 13
# ----------------------------------------------------------------------

"""
Usage:
    python train.py --episodes 1000
"""

import gym

from rltools.utils.callbacks import ModelSaver, TrainingLogger
from rltools.utils import GymWrapper

from common import display_versions, initial_seed, timer
import config

# ----------------------------------------------------------------------
def train_agent(env, mode="dqn"):
    state_size, n_actions = env.info()

    if mode == "dqn":
        from dqn import CartAgent, training_loop
        agent = CartAgent(state_size, n_actions)
    elif mode == "evo":
        from evo import CartAgent, training_loop
        agent = CartAgent(state_size, n_actions)
    elif mode == "pg":
        #from policygrad import PongAgent, training_loop
        pass

    callbacks = [
        TrainingLogger("training.log"),
        #ModelSaver(agent, config.agent_file),
    ]

    agent = training_loop(agent, env, config, callbacks)
    return agent

# ----------------------------------------------------------------------
def main():
    config.read_args()

    initial_seed(config.seed)
    display_versions()

    gymenv = gym.make("CartPole-v1")
    gymenv._max_episode_steps = 2000
    env = GymWrapper(gymenv, seed=config.seed, render=False)

    agent = train_agent(env, "evo")
    agent.save(config.agent)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    with timer("CartPole train"):
        main()