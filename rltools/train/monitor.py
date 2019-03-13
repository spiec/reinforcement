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
import matplotlib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
class MonitorBase(object):

    def __init__(self):
        super(MonitorBase, self).__init__()
        
        matplotlib.rcParams.update({"font.size": 10})
        plt.ion()
        self.fig = plt.figure(figsize=(20, 10))

    def wait(self):
        plt.ioff()
        plt.show()

    def redraw(self):
        self.fig.canvas.draw()

# ----------------------------------------------------------------------
class TrainingMonitor(MonitorBase):

    def __init__(self):
        super(TrainingMonitor, self).__init__()

        self.axes = []
        nrows = 1
        ncols = 2
        for i in range(nrows * ncols):
            self.axes.append(self.fig.add_subplot(nrows, ncols, i + 1))

        self._rewards = []
        self._rewards_to_plot = []
        self._epsilons = []
        self._n_last = 50

    def episode_finished(self, params):
        # status unpacking
        episode_idx, reward, agent_epsilon = params

        print("===== Episode FINISHED, reward: {} =====".format(reward))
        self._rewards.append(reward)

        if episode_idx % self._n_last == 0 and (episode_idx > 0):
            mean_reward = np.mean(self._rewards[-self._n_last:])
            self._rewards_to_plot.append(mean_reward)

            self._plot_rewards(self.axes[0])
            self._epsilons.append(agent_epsilon)
            self._plot_epsilon(self.axes[1])
            self.redraw()
            sys.stderr.write("Mean reward: {:.2f} (episode: {}) \n".format(mean_reward,
                                                                           episode_idx))

    def _plot_rewards(self, axis):
        axis.clear()
        axis.set_title("Mean reward (last {} episodes)".format(self._n_last))
        axis.plot(self._rewards_to_plot)

    def _plot_epsilon(self, axis):
        axis.clear()
        axis.set_title("Agent epsilon")
        axis.plot(self._epsilons)