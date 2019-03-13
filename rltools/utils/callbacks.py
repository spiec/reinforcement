#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 10
# ----------------------------------------------------------------------

"""
"""

# ----------------------------------------------------------------------
class TrainingLogger(object):

    def __init__(self, filename):
        self.file = open(filename, "w")

    def __call__(self, info):
        line = ",".join([str(v) for v in info]) + "\n"
        self.file.write(line)
        self.file.flush()

    def __del__(self):
        self.file.close()

# ----------------------------------------------------------------------
class ModelSaver(object):

    def __init__(self, agent, filename):
        self.agent = agent
        self.filename = filename

    def __call__(self, info):
        self.agent.save(self.filename)