# -*- coding: utf-8 -*-
import functools
import time

"""
Collection of small utility functions used across the project
"""

def timer(f):
    """ Add this @decorator to a function to print its runtime after completion """
    @functools.wraps(f)
    def t_wrap(*args, **kwargs):
        t_start = time.perf_counter_ns()
        ret = f(*args, **kwargs)
        t_run = time.perf_counter_ns() - t_start
        print(f"{f.__name__} completed in {t_run} seconds.")
        return ret
    return t_wrap