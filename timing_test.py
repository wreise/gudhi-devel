# Author(s): Wojciech Reise
#
# Copyright (C) 2022 Inria
import json
import os
import resource
import time
from itertools import product

import numpy as np
import psutil as psutil
from gudhi.representations.silhouettes import Silhouette, SilhouetteNumpy, SilhouetteKeops
from tqdm import tqdm

from src.python.gudhi.representations.landscapes import Landscape, LandscapeNumpy, LandscapeKeops


def pow(n):
  return lambda x: np.power(x[1]-x[0], n)


def get_diagrams(n_dgms, n_points=1000, concentrate_in_band=None):
    """Generate a collection of diagrams, each with n_points. `concentrate_in_band`
    can be used to put all but one point close to the diagonal, in a band
    whose width is defined by that parameter.
    Params:
        n_dgms, n_points: ints
        concentrate_in_band: positive float
    """
    if concentrate_in_band:
        births = np.random.rand(n_dgms, n_points, 1)
        persistences = concentrate_in_band*np.random.rand(n_dgms, n_points, 1)
        births[:, 0, 0], persistences[:, 0, 0] = 0., 2.
        dgms = np.concatenate([births, births+persistences], axis=2)
    else:
        dgms = np.random.rand(n_dgms, n_points, 2)
        dgms[..., 1] += dgms[..., 0]
    return dgms


def round_floats(o):
    #https://stackoverflow.com/a/53798633/15150356
    if isinstance(o, float): return round(o, 5)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

def elapsed_since(start):
    #return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    return time.time() - start
def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss
def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        return elapsed_time, mem_after - mem_before
    return wrapper

@track
def calculate_silhouette(algo, dgms):
    slt = algo(resolution=1000, weight=pow(2))
    slt.fit_transform(dgms)
    return dgms

@track
def calculate_landscape(algo, dgms):
    lds = algo(num_landscapes=5, resolution=1000, sample_range=[np.nan, np.nan])
    lds.fit_transform(dgms)
    return dgms

if __name__ == "__main__":

    n_realizations = 10
    n_dgms = 20
    n_points = np.logspace(np.log10(100), np.log10(10000), 6).astype(int)
    #n_points = np.logspace(np.log10(10), np.log10(100), 2).astype(int)
    concentrate = np.array([*np.logspace(-3, np.log10(1.), 5)])
    silhouette_algo = {
        "current": Silhouette, "numpy": SilhouetteNumpy, "keops": SilhouetteKeops,
    }
    landscape_algo = {
        "current": Landscape, "numpy": LandscapeNumpy, "keops": LandscapeKeops,
    }

    results, total = [], len(n_points)*n_realizations*len(concentrate)
    for n_p, c, n_r in tqdm(product(n_points, concentrate, range(n_realizations)), total=total):
        dgms = get_diagrams(n_dgms, n_p, concentrate_in_band=c)
        for algo_name, algo in landscape_algo.items():
            exec_time, memory_print = calculate_landscape(algo, dgms)

            results.append({"n_points": n_p, "concentrate": -1 if c is None else c,
                            "realization": n_r, "algo": algo_name,
                            "time": exec_time, "memory": memory_print})

    with open("./landscaes_timing_results.json", "w") as f:
        json.dump(round_floats(results), f)
