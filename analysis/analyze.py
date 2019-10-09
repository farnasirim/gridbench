#!/usr/bin/env python
import os
import sys
import numpy as np
import json
import collections
import math
import pickle

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import pylab as pl
from matplotlib import collections  as mc

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import color


def select(results, key):
    ret = [getattr(result, key) if not callable(getattr(result, key)) else
           getattr(result, key)() for result in results]
    return ret


def where(results, filter_func):
    return [result for result in results if filter_func(result)]


class Result:
    def __init__(self, para, core, skew,
                 insert_time, init_time, exec_time):
        self.para = para
        self.core = core
        self.skew = skew

        self.insert_time = insert_time
        self.init_time = init_time
        self.exec_time = exec_time

    def total_time(self):
        return self.insert_time + self.init_time + self.exec_time

    def __repr__(self):
        d = {}
        good_keys = ["para", "core", "skew", "insert_time", "init_time", "exec_time"]
        for key in good_keys:
            d[key] = getattr(self, key)

        return json.dumps(d)


def result_from_dir(dir):
    try:
        with open(os.path.join(dir, "meta")) as f:
            meta = json.load(f)
            handle_parallel = meta["VHandleParallel"]
            core_scaling = meta["CoreScaling"]
            skew = meta["YcsbSkewFactor"]
        with open(os.path.join(dir, "felis.out")) as f:
            lines = f.readlines()
            for line in lines:
                if "Insert / Initialize / Execute" in line:
                    splitted_line = line.split(' ')
                    ints = []
                    for part in splitted_line:
                        try:
                            ints.append(int(part))
                        except:
                            pass
                    ins, ini, exe = ints[-3:]
                    break

        return Result(handle_parallel, core_scaling, skew,
                      insert_time=ins, init_time=ini, exec_time=exe)
    except:
        pass
    return None


def main():
    cwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    runs_data_dir = os.path.join(cwd, sys.argv[1])
    maybe_result_dirs = os.listdir(runs_data_dir)
    maybe_result_dirs = list(map(lambda p: os.path.join(runs_data_dir, p),
                                 maybe_result_dirs))


    with ThreadPool(10) as p:
        results = p.map(result_from_dir, maybe_result_dirs)
        results = list(filter(lambda x: x is not None, results))


    skews = collections.defaultdict(list)

    for result in results:
        skews[result.skew].append(result)


    for skew in skews.keys():
        current_results = skews[skew]
        plot_grid(xs=select(current_results, "para"),
                  ys=select(current_results, "core"),
                  vals=select(current_results, "total_time"),
                  title="skew: {}".format(skew),
                  xlabel="VHandleParallel",
                  ylabel="CoreScaling")

    plt.show()


def plot_grid(xs, ys, vals, title, xlabel="", ylabel=""):
    fig, ax = plt.subplots()

    d = 5
    xmin = min(0, min(xs)) - d
    xmax = max(100, max(xs)) + d
    ymin = min(0, min(ys)) - d
    ymax = max(100, max(ys)) + d

    print(int(xmin), int(xmax), int(ymin), int(ymax))
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    smallest_val = min(vals)
    biggest_val = max(vals)
    value_range = biggest_val - smallest_val

    smallest_circle = 0.1
    biggest_circle = 2
    circle_range = value_range

    color_range = value_range
    start_color = color.rgb_to_hsv(0, 1, 0)
    end_color = color.rgb_to_hsv(1, 0, 0)

    for (x, y, val) in zip(xs, ys, vals):
        current_color = color.hsv_to_rgb(*color.lerp3(start_color, end_color, val - smallest_val, color_range))
        current_radius = smallest_circle + (val - smallest_val) * biggest_circle / value_range
        ax.add_artist(plt.Circle((x, y), current_radius, color=current_color))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)


if __name__ == "__main__":
    main()
