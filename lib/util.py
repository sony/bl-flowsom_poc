# Copyright 2025 Sony Corporation
#
# BL-FlowSOM PoC source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pandas as pd


class BatchSOM:
    def __init__(self, grid_size, data, beta=0.33, sigma_pca=5):
        """
        Initialize the BatchSOM object.

        Parameters:
        - grid_size: tuple (X, Y), the size of the SOM grid (X columns, Y rows).
        - data: numpy array of shape (n_samples, n_features), the input data.
        - beta: float algorithm parameter.
        - sigma_pca: float algorithm parameter.
        """
        self.grid_size = grid_size
        self.sigma_pca = sigma_pca
        self.grid_pts = self._create_grid(*grid_size)
        self.beta = beta
        self.codes = self._init_code_by_pca(data, xdim=grid_size[0], ydim=grid_size[1])
        nhbrdist = np.max(
            np.abs(self.grid_pts[:, np.newaxis] - self.grid_pts[np.newaxis, :]), axis=2
        )
        self.radius = np.quantile(nhbrdist, beta) * np.array([1, 0])

        # radii is determined after with rlen
        self.radii = None

    def _init_code_by_pca(self, data, xdim=10, ydim=10):
        """
        Calculate initial vector of each code using PCA

        Parameters:
        - data: numpy array of shape (n_samples, n_features), the input data.
        - xdim: the size of the SOM grid for X (X columns, Y rows).
        - ydim: the size of the SOM grid for Y (X columns, Y rows).
        - sigma_pca:algorythm parameter
        """

        n_codes = xdim * ydim
        ave_vec = np.mean(data, axis=0)
        pca = PCA()
        pca.fit(data)

        pc1_vec = pca.components_[0]
        pc2_vec = pca.components_[1]

        if np.max(pc1_vec) + np.min(pc1_vec) < 0:
            pc1_vec = -pc1_vec
        if np.max(pc2_vec) + np.min(pc2_vec) < 0:
            pc2_vec = -pc2_vec
        sigma_pc1 = pca.transform(data)[:, 0].std()
        sigma_pc2 = pca.transform(data)[:, 1].std()

        codes = np.zeros((n_codes, data.shape[1]))

        for y in range(ydim):
            for x in range(xdim):
                # memo
                # x -> x+1
                # y -> y+1
                codes[y * xdim + x] = ave_vec + (
                    self.sigma_pca * sigma_pc1 / xdim * pc1_vec * ((x + 1) - xdim / 2)
                    + self.sigma_pca * sigma_pc2 / ydim * pc2_vec * ((y + 1) - ydim / 2)
                )
        return codes

    def _create_grid(self, xdim, ydim):
        """
        Create a grid of points for the SOM.

        Parameters:
        - xdim: int, number of columns in the grid.
        - ydim: int, number of rows in the grid.

        Returns:
        - numpy array of shape (x * y, 2), the grid points in 2D space.
        """
        x, y = np.meshgrid(np.arange(xdim), np.arange(ydim))
        return np.column_stack([x.ravel(), y.ravel()])

    def _set_radii(self, rlen=10):
        """
        Set radii with rlen

        Parameters:
        - rlen: int, number of learning iteration.

        Returns:
        - None
        """
        self.radii = np.linspace(self.radius[0], self.radius[1], rlen)

    def _pairwise_distance(self, points1, points2):
        """
        Calculate pairwise Euclidean distances between two sets of points.

        Parameters:
        - points1: numpy array of shape (n1, d), first set of points.
        - points2: numpy array of shape (n2, d), second set of points.

        Returns:
        - numpy array of shape (n1, n2), pairwise distances.
        """
        diff = points1[:, None, :] - points2[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    def train(self, data, rlen=10):
        """
        Train the SOM using batch learning.

        Parameters:
        - data: numpy array of shape (n_samples, n_features), the input data.
        - rlen: int nuber of learning iteration

        Returns:
        - self: Trained BatchSOM instance.
        """

        self._set_radii(rlen=rlen)

        data = np.asarray(data)
        nd, nf = data.shape  # Number of data points and features
        ng = self.grid_pts.shape[0]  # Number of grid points

        # Initialize SOM codes if not provided
        if self.codes is None:
            self.codes = data[np.random.choice(nd, ng, replace=False), :]

        # Precompute pairwise distances between grid points
        nhbrdist = self._pairwise_distance(self.grid_pts, self.grid_pts)

        # Batch learning iterations
        for r in self.radii:
            # Find nearest grid point for each data point
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self.codes)
            cl = nn.kneighbors(data, return_distance=False).flatten()

            # Create adjacency matrix based on neighborhood radius
            A = (nhbrdist <= r).astype(int)[:, cl]
            # Update grid codes
            ind = A.sum(axis=1) > 0
            if np.any(ind):
                numerator = (
                    A[ind, :] @ data
                )  # Weighted sum of data allocated to the target node
                denominator = A[ind, :].sum(axis=1)[:, None]  # number of events
                self.codes[ind, :] = numerator / denominator  # update new weight

        return self

    def fit(self, data):
        """
        Assign data points to the closest SOM nodes.

        Parameters:
        - data: numpy array of shape (n_samples, n_features), the input data.

        Returns:
        - pandas DataFrame of shape (n_samples,), the indices of the closest grid points for each data point.
        """
        data = np.asarray(data)

        # Find nearest grid point for each data point
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.codes)
        cl = nn.kneighbors(data, return_distance=False).flatten()
        cl = cl + 1  # 0 origin -> 1 origin
        pdcl = pd.DataFrame(cl)
        return pdcl

    def get_codes(self):
        """
        Get the trained SOM codes.

        Returns:
        - pandas Dataframe of shape (X * Y, n_features), the trained SOM codes.
        """
        pdcode = pd.DataFrame(self.codes)
        return pdcode

    def get_grid(self):
        """
        Get the SOM grid points.

        Returns:
        - numpy array of shape (X * Y, 2), the SOM grid points.
        """
        return self.grid_pts
