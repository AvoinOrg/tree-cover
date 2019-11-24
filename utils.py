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
        t_start = time.perf_counter()
        ret = f(*args, **kwargs)
        t_run = round((time.perf_counter() - t_start)/60)
        print(f"{f.__name__} completed in {t_run} minutes.")
        return ret
    return t_wrap