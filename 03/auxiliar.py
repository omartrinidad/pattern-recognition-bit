#!/usr/bin/env python
# encoding: utf8

"""
Auxiliar functions with dirty tricks to deal with plots, timing, etc.
"""

from matplotlib2tikz import save as tikz_save
import time


def measure_time(func):
    """add time measure decorator to the functions"""
    def func_wrapper(*args, **kwargs):
        start_time = time.time()
        a = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time
    return func_wrapper


def save_figure():
    """save figure using only a decorator"""
    def decorator(function):
        def wrapper(*args, **kwargs):
            plt = function(*args, **kwargs)
            path = kwargs["path"]
            if path.endswith(".tex"):
                tikz_save(path)
            else:
                plt.savefig(path, bbox_inches="tight", pad_inches=0)
            plt.close()
        return wrapper
    return decorator
