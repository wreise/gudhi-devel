# Author(s): Wojciech Reise
#
# Copyright (C) 2022 Inria
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.python.gudhi.representations.vector_methods import _automatic_sample_range


class Landscape(BaseEstimator, TransformerMixin):
    """
    This is a class for computing persistence landscapes from a list of persistence diagrams. A persistence landscape is a collection of 1D piecewise-linear functions computed from the rank function associated to the persistence diagram. These piecewise-linear functions are then sampled evenly on a given range and the corresponding vectors of samples are concatenated and returned. See http://jmlr.org/papers/v16/bubenik15a.html for more details.
    """

    def __init__(self, num_landscapes=5, resolution=100, sample_range=[np.nan, np.nan]):
        """
        Constructor for the Landscape class.

        Parameters:
            num_landscapes (int): number of piecewise-linear functions to output (default 5).
            resolution (int): number of sample for all piecewise-linear functions (default 100).
            sample_range ([double, double]): minimum and maximum of all piecewise-linear function domains, of the form [x_min, x_max] (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, it can be computed from the persistence diagrams with the fit() method.
        """
        self.num_landscapes, self.resolution, self.sample_range = num_landscapes, resolution, sample_range
        self.nan_in_range = np.isnan(np.array(self.sample_range))
        self.new_resolution = self.resolution + self.nan_in_range.sum()

    def fit(self, X, y=None):
        """
        Fit the Landscape class on a list of persistence diagrams: if any of the values in **sample_range** is numpy.nan, replace it with the corresponding value computed on the given list of persistence diagrams.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        self.sample_range = _automatic_sample_range(np.array(self.sample_range), X, y)
        return self

    def transform(self, X):
        """
        Compute the persistence landscape for each persistence diagram individually and concatenate the results.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.

        Returns:
            numpy array with shape (number of diagrams) x (number of samples = **num_landscapes** x **resolution**): output persistence landscapes.
        """
        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.new_resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            ls = np.zeros([self.num_landscapes, self.new_resolution])

            events = []
            for j in range(self.new_resolution):
                events.append([])

            for j in range(num_pts_in_diag):
                [px, py] = diagram[j, :2]
                min_idx = np.clip(np.ceil((px - self.sample_range[0]) / step_x).astype(int), 0, self.new_resolution)
                mid_idx = np.clip(np.ceil((0.5 * (py + px) - self.sample_range[0]) / step_x).astype(int), 0,
                                  self.new_resolution)
                max_idx = np.clip(np.ceil((py - self.sample_range[0]) / step_x).astype(int), 0, self.new_resolution)

                if min_idx < self.new_resolution and max_idx > 0:

                    landscape_value = self.sample_range[0] + min_idx * step_x - px
                    for k in range(min_idx, mid_idx):
                        events[k].append(landscape_value)
                        landscape_value += step_x

                    landscape_value = py - self.sample_range[0] - mid_idx * step_x
                    for k in range(mid_idx, max_idx):
                        events[k].append(landscape_value)
                        landscape_value -= step_x

            for j in range(self.new_resolution):
                events[j].sort(reverse=True)
                for k in range(min(self.num_landscapes, len(events[j]))):
                    ls[k, j] = events[j][k]

            if self.nan_in_range[0]:
                ls = ls[:, 1:]
            if self.nan_in_range[1]:
                ls = ls[:, :-1]
            ls = np.sqrt(2) * np.reshape(ls, [1, -1])
            Xfit.append(ls)

        Xfit = np.concatenate(Xfit, 0)

        return Xfit

    def __call__(self, diag):
        """
        Apply Landscape on a single persistence diagram and outputs the result.

        Parameters:
            diag (n x 2 numpy array): input persistence diagram.

        Returns:
            numpy array with shape (number of samples = **num_landscapes** x **resolution**): output persistence landscape.
        """
        return self.fit_transform([diag])[0, :]

class LandscapeNumpy(Landscape):

    def transform(self, X):
        Xfit = []
        x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.new_resolution)
        for i, diag in enumerate(X):
            midpoints, heights = (diag[:, 0] + diag[:, 1]) / 2., (diag[:, 1] - diag[:, 0]) / 2.
            tent_functions = np.maximum(heights[None, :] - np.abs(x_values[:, None] - midpoints[None, :]), 0)
            tent_functions.partition(diag.shape[0] - self.num_landscapes, axis=1)
            landscapes = np.sort(tent_functions[:, -self.num_landscapes:], axis=1)[:, ::-1].T

            if self.nan_in_range[0]:
                landscapes = landscapes[:, 1:]
            if self.nan_in_range[1]:
                landscapes = landscapes[:, :-1]
            landscapes = np.sqrt(2) * np.ravel(landscapes)
            Xfit.append(landscapes)

        return np.stack(Xfit, axis=0)

class LandscapeKeops(Landscape):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        landscape_formula = "(-1)*ReLU(heights - Abs(x_values - midpoints))"
        variables = [
            "heights = Vi(1)",
            "midpoints = Vi(1)",
            "x_values = Vj(1)",
        ]
        from pykeops.numpy import Genred
        self.landscape = Genred(landscape_formula, variables, reduction_op="KMin",
                                axis=0, opt_arg=self.num_landscapes)

    def transform(self, X):
        landscapes_list = []
        x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.new_resolution)

        for i, diag in enumerate(X):
            midpoints, heights = (diag[:, 0] + diag[:, 1]) / 2., (diag[:, 1] - diag[:, 0]) / 2.
            landscapes = (-1) * self.landscape(heights[:, None], midpoints[:, None], x_values[:, None]).T

            if self.nan_in_range[0]:
                landscapes = landscapes[:, 1:]
            if self.nan_in_range[1]:
                landscapes = landscapes[:, :-1]
            landscapes = np.sqrt(2) * np.ravel(landscapes)
            landscapes_list.append(landscapes)

        return np.stack(landscapes_list, axis=0)
