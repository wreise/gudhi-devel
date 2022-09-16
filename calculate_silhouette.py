# Author(s): Wojciech Reise
#
# Copyright (C) 2022 Inria
import argparse
import os
import time

import numpy as np
import psutil


def calculate_silhouette(algo, dgms):
    slt = algo(resolution=1000, weight=lambda x: (x[1]-x[0])**2)
    slt.fit_transform(dgms)
    return dgms


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("-diagramfile", type=str)
    args.add_argument("-algoname", type=str)
    args = args.parse_args()

    diagram_file, algo_name = args.diagramfile, args.algoname
    dgms = np.load(diagram_file)

    start = time.time()
    if algo_name == "cpp":
        from gudhi.representations.silhouettes import SilhouetteCPP
        slt = SilhouetteCPP(2., 1., resolution=1000)
        slt.fit_transform(dgms)
    else:
        if algo_name == "current":
            from gudhi.representations.silhouettes import Silhouette
            algo = Silhouette
        elif algo_name == "numpy":
            from gudhi.representations.silhouettes import SilhouetteNumpy
            algo = SilhouetteNumpy
        elif algo_name == "keops":
            from gudhi.representations.silhouettes import SilhouetteKeops
            algo = SilhouetteKeops

        calculate_silhouette(algo, dgms)

    memory_used = psutil.Process(os.getpid()).memory_info().rss
    elapsed_time = time.time() - start

    print(' $', elapsed_time, '$', memory_used, '$ ')
