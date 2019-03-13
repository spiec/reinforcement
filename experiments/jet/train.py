#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 11
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
from environment import JetEnvironment

# ----------------------------------------------------------------------
def train_agent(env, mode="dqn"):
    state_size, n_actions = env.info()

    if mode == "dqn":
        from dqn import JetAgent, training_loop
        agent = JetAgent(state_size, n_actions)
    elif mode == "evo":
        from evo import JetAgent, training_loop
        agent = JetAgent(state_size, n_actions)
    elif mode == "pg":
        #from policygrad import PongAgent, training_loop
        pass

    callbacks = [
        TrainingLogger("logs/tmp.log"),
        ModelSaver(agent, config.agent), #best_only=True, intervals=[]),
    ]

    agent = training_loop(agent, env, config, callbacks)
    return agent

# ----------------------------------------------------------------------
def main():
    config.read_args()

    initial_seed(config.seed)
    display_versions()

    env = JetEnvironment(render=False)
    env.reset()

    agent = train_agent(env, "dqn")
    agent.save(config.agent)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    with timer("Jet train"):
        main()