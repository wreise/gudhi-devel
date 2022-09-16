# Author(s): Wojciech Reise
#
# Copyright (C) 2022 Inria
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.python.gudhi.representations.vector_methods import _automatic_sample_range
from .silhouette import silhouette_cpp


class Silhouette(BaseEstimator, TransformerMixin):
    """
    This is a class for computing persistence silhouettes from a list of persistence diagrams. A persistence silhouette is computed by taking a weighted average of the collection of 1D piecewise-linear functions given by the persistence landscapes, and then by evenly sampling this average on a given range. Finally, the corresponding vector of samples is returned. See https://arxiv.org/abs/1312.0308 for more details.
    """

    def __init__(self, weight=lambda x: 1, resolution=100, sample_range=[np.nan, np.nan]):
        """
        Constructor for the Silhouette class.

        Parameters:
            weight (function): weight function for the persistence diagram points (default constant function, ie lambda x: 1). This function must be defined on 2D points, ie on lists or numpy arrays of the form [p_x,p_y].
            resolution (int): number of samples for the weighted average (default 100).
            sample_range ([double, double]): minimum and maximum for the weighted average domain, of the form [x_min, x_max] (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, it can be computed from the persistence diagrams with the fit() method.
        """
        self.weight, self.resolution, self.sample_range = weight, resolution, sample_range
        self.im_range = None

    def fit(self, X, y=None):
        """
        Fit the Silhouette class on a list of persistence diagrams: if any of the values in **sample_range** is numpy.nan, replace it with the corresponding value computed on the given list of persistence diagrams.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        self.sample_range = _automatic_sample_range(np.array(self.sample_range), X, y)
        self.im_range = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        return self

    def transform(self, X):
        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]

            sh, weights = np.zeros(self.resolution), np.zeros(num_pts_in_diag)
            for j in range(num_pts_in_diag):
                weights[j] = self.weight(diagram[j, :])
            total_weight = np.sum(weights)

            for j in range(num_pts_in_diag):

                [px, py] = diagram[j, :2]
                weight = weights[j] / total_weight
                min_idx = np.clip(np.ceil((px - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)
                mid_idx = np.clip(np.ceil((0.5 * (py + px) - self.sample_range[0]) / step_x).astype(int), 0,
                                  self.resolution)
                max_idx = np.clip(np.ceil((py - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)

                if min_idx < self.resolution and max_idx > 0:

                    silhouette_value = self.sample_range[0] + min_idx * step_x - px
                    for k in range(min_idx, mid_idx):
                        sh[k] += weight * silhouette_value
                        silhouette_value += step_x

                    silhouette_value = py - self.sample_range[0] - mid_idx * step_x
                    for k in range(mid_idx, max_idx):
                        sh[k] += weight * silhouette_value
                        silhouette_value -= step_x

            Xfit.append(np.reshape(np.sqrt(2) * sh, [1, -1]))

        Xfit = np.concatenate(Xfit, 0)
        return Xfit

    def __call__(self, diag):
        """
        Apply Silhouette on a single persistence diagram and outputs the result.

        Parameters:
            diag (n x 2 numpy array): input persistence diagram.

        Returns:
            numpy array with shape (**resolution**): output persistence silhouette.
        """
        return self.fit_transform([diag])[0, :]


class SilhouetteNumpy(Silhouette):

    def transform(self, X):
        Xfit = []
        x_values = self.im_range

        for i, diag in enumerate(X):
            midpoints, heights = (diag[:, 0] + diag[:, 1]) / 2., (diag[:, 1] - diag[:, 0]) / 2.
            weights = np.array([self.weight(point) for point in diag])
            total_weight = np.sum(weights)

            tent_functions = np.maximum(heights[None, :] - np.abs(x_values[:, None] - midpoints[None, :]), 0)
            silhouette = np.sum(weights[None, :] / total_weight * tent_functions, axis=1)
            Xfit.append(silhouette * np.sqrt(2))

        return np.stack(Xfit, axis=0)


class SilhouetteKeops(Silhouette):

    def __init__(self, weight=lambda x: 1, resolution=100, sample_range=[np.nan, np.nan]):
        """
        Constructor for the Silhouette class.

        Parameters:
            weight (function): weight function for the persistence diagram points (default constant function, ie lambda x: 1). This function must be defined on 2D points, ie on lists or numpy arrays of the form [p_x,p_y].
            resolution (int): number of samples for the weighted average (default 100).
            sample_range ([double, double]): minimum and maximum for the weighted average domain, of the form [x_min, x_max] (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, it can be computed from the persistence diagrams with the fit() method.
        """
        super().__init__(weight, resolution, sample_range)

        silhouette_formula = "normalized_weights * ReLU(heights - Abs(x_values - midpoints))"
        variables = [
            "normalized_weights = Vi(1)",
            "heights = Vi(1)",
            "midpoints = Vi(1)",
            "x_values = Vj(1)",
        ]
        from pykeops.numpy import Genred
        self.silhouette = Genred(silhouette_formula, variables, reduction_op="Sum", axis=0)

    def transform(self, X):
        """
        Compute the persistence silhouette for each persistence diagram individually and concatenate the results.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.

        Returns:
            numpy array with shape (number of diagrams) x (**resolution**): output persistence silhouettes.
        """
        silhouettes_list = []
        x_values = self.im_range
        for i, diag in enumerate(X):
            midpoints, heights = (diag[:, 0] + diag[:, 1]) / 2., (diag[:, 1] - diag[:, 0]) / 2.
            weights = np.array([self.weight(point) for point in diag])
            weights /= np.sum(weights)

            silhouettes_list.append(
                np.sqrt(2) * self.silhouette(weights[:, None], heights[:, None],
                                             midpoints[:, None], x_values[:, None])[:, 0]
            )

        return np.stack(silhouettes_list, axis=0)

class SilhouetteCPP(Silhouette):

    def __init__(self, p, epsilon, resolution=100, sample_range=[np.nan, np.nan]):
        super().__init__(lambda x: max(x[1]-x[0]-epsilon, 0)**p, resolution, sample_range)
        self.p = p
        self.epsilon = p

    def fit(self, X, y=None):
        self.sample_range = _automatic_sample_range(np.array(self.sample_range), X, y)
        self.im_range = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        return self

    def transform(self, X):
        result = np.stack([
            silhouette_cpp(diag, self.p, self.epsilon, self.sample_range, self.resolution)
            for diag in X
        ])
        return result

