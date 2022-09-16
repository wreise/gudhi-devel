# Author(s): Wojciech Reise
#
# Copyright (C) 2022 Inria
import json
import subprocess
from itertools import product

import numpy as np
from tqdm import tqdm


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
        births[:, 0, 0], persistences[:, 0, 0] = 0., 1.
        dgms = np.concatenate([births, births+persistences], axis=2)
    else:
        dgms = np.random.rand(n_dgms, n_points, 2)
        dgms[..., 1] += dgms[..., 0]
    return dgms


path = "/home/wreise/Libs/gudhi-devel/build/src/python/"
env = {"PYTHONPATH": path,
       "cwd": "/home/wreise/Libs/gudhi-devel/"}
python_exec = "/home/wreise/snap/miniconda3/envs/gudhi/bin/python"
if __name__ == "__main__":

    n_realizations = 10
    n_dgms = 20
    n_points = np.logspace(np.log10(100), np.log10(10000), 5).astype(int)
    #n_points = np.logspace(np.log10(2), np.log10(10), 2).astype(int)
    #n_points = np.logspace(np.log10(10), np.log10(100), 2).astype(int)
    concentrate = np.array([None, *np.logspace(-3, np.log10(0.5), 4)])
    silhouette_algo = ["current", "numpy", "keops", "cpp"]

    n_repeat = 5

    results, total = [], len(n_points)*n_realizations*len(concentrate)
    for n_p, c, n_r in tqdm(product(n_points, concentrate, range(n_realizations)), total=total):
        dgms = get_diagrams(n_dgms, n_p, concentrate_in_band=c)
        np.save("dgm.npy", dgms)
        for algo_name in silhouette_algo:
            completed = subprocess.run([python_exec, "calculate_silhouette.py",
                                        "-diagramfile=dgm.npy",
                                        f"-algoname={algo_name}"],
                                       capture_output=True,
                                       env=env)
            output = str(completed.stdout)
            _, exec_time, memory, _ = output.split('$')
            results.append({"n_points": int(n_p), "concentrate": -1 if c is None else c,
                            "realization": n_r, "algo": algo_name,
                            "time": exec_time, "memory": int(memory)})

    with open("./timing_results.json", "w") as f:
        json.dump(results, f)