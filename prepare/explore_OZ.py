#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:54:15 2019
Visualises the seasonal pattern of the interesting band values over the course of a year
group by tree cover.
@author: fynn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .join_GEE import join_data_by_location

plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams["font.size"] = 20

file_path = "data/export_2015_all_australia.csv"
orig_data = "data/by_region/region_Australia.csv"

orig_data = pd.read_csv(orig_data)
data = pd.read_csv(file_path)
data["date"] = pd.DatetimeIndex(data["date"])

joint = join_data_by_location(data, orig_data)

# counts per tree cover level:
print(joint.groupby("tree_cover")["tree_cover"].count())
# 0.00    227569
# 0.02     24087
# 0.04     16806
# 0.06     14449
# 0.08     22446
# 0.15     30050
# 0.25     15832
# 0.35      7193
# 0.45      3666
# 0.55      2167
# 0.65      2399
# 0.75      1797
# 0.85      1780
# 0.95      2250

joint["date"] = pd.DatetimeIndex(joint["date"]).normalize()
joint.groupby("tree_cover").plot(x="date", y=["B2", "B3", "B4"])
plt.show()

color_dict = {"B2": "blue", "B3": "green", "B4": "red", "B5": "grey"}
cols = ["B2", "B3", "B4", "B5"]
for name, group in joint.groupby(["tree_cover"]):
    g_agg = group.groupby("date").aggregate(np.median)
    ax = g_agg.plot(y=cols, title=f"tree cover: {name}", color=[color_dict[x] for x in cols])
    ax.set_ylim(0, 4000)
    plt.savefig(f"img/OZ_2015_{name}_tree_cover_per_day.png")
    plt.show()
