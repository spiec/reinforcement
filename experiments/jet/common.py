#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 13
# ----------------------------------------------------------------------

"""Re-usable set of common utilities.
"""

from datetime import datetime

import os
import random
import time
import sys

from contextlib import contextmanager
from termcolor import colored

import joblib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

# ----------------------------------------------------------------------
@contextmanager
def timer(name):
    t0 = time.time()
    print("Starting [{}] ({})".format(name, datetime.now()))
    yield
    print("[{0}] done in {1:.2f} [s]".format(name, time.time() - t0))

# ----------------------------------------------------------------------
def timeit(method):
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        print("[{0}] {1:.6f} [s]".format(method.__name__, end - start))
        return result
    
    return timed

# ----------------------------------------------------------------------
def initial_seed(seed):
    """Set initial seeds for all used modules.
    """
    print("Initial seed {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ----------------------------------------------------------------------
def backup_run(copy_list, target_dir="./runs"):
    """Saves all relevant to the current experiment/run files.
    """
    print("Backup current run ({})".format(target_dir))

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(target_dir, run_stamp)
    
    for directory in copy_list:
        target = os.path.join(run_dir,
                              directory.strip("..").strip("/").strip("..").strip("/"))          # TODO
        print("Copying", directory, "to", target)
        
        if os.path.isfile(directory):
            shutil.copy(directory, run_dir)
        else:
            shutil.copytree(directory, target,
                            ignore=shutil.ignore_patterns(".pyc", "__pycache__"))
    # optionally gzip
    
    return os.path.join(target_dir, run_stamp)

# ----------------------------------------------------------------------
class RunLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        """This flush method is needed for python 3 compatibility.
        """

# ----------------------------------------------------------------------
def save_model(model, filename):
    joblib.dump(model, filename)

# ----------------------------------------------------------------------
def load_model(filename):
    return joblib.load(filename)

# ----------------------------------------------------------------------
def display_versions():
    """Displays versions of standard DS software stack.
    """
    import importlib
    libs = ["numpy", "pandas", "sklearn",
            "lightgbm", "xgboost", "catboost",
            "keras", "tensorflow",
            "torch"]

    for lib in libs:
        module = importlib.import_module(lib)
        print("{}: {}".format(lib, module.__version__))

# ----------------------------------------------------------------------
def color_print(color, txt, *args, **kwargs):
    print(_str_to_color(color) + txt + _str_to_color("endc"), *args, **kwargs)

# ----------------------------------------------------------------------
def _str_to_color(color):
    color = color.lower()

    if color == "header": return '\033[95m'
    elif color == "blue": return '\033[94m'
    elif color == "green": return '\033[92m'
    elif color == "warn": return '\033[93m'
    elif color == "fail": return '\033[91m'
    elif color == "bold": return '\033[1m'
    elif color == "underline": return '\033[4m'
    elif color == "endc": return '\033[0m'
    return '\033[0m'        # end color

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pass