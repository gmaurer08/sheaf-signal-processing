""" builder.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

from dataclasses import dataclass

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import gstools as gs
import pyvista as pv

@dataclass
class CochainSample:
    """ Wrapper for a cochain sample
    """
    X: np.ndarray
    covariance: np.ndarray
    X_GT: np.ndarray
    
    def random_tangent_bundle_signals(
            self, 
            Sigma : np.ndarray = None, 
            len_scale: float = 10,
            M : int = 1000, 
            seed : int = 42
        ) -> None:

        if Sigma is None:
            Sigma = np.eye(self.d)

        # Create RNG
        rng = np.random.default_rng(seed)

        # Build mesh
        mesh = pv.PolyData(self.points)

        # Model for spatial correlation
        model = gs.Gaussian(dim=3, var=1.0, len_scale=len_scale)
        srf = gs.SRF(model, mean=(0, 0, 0), generator="VectorField")

        # Cholesky
        L = np.linalg.cholesky(Sigma)
        vec_fields = np.empty((M, self.points.shape[0], 3))

        for m in range(M):
            # Use the same RNG for SRF
            f = srf.mesh(mesh, points="points", seed=rng.integers(1e9))
            vec_fields[m] = (L @ f).T

        vec_fields = vec_fields.reshape(M, 3 * self.points.shape[0])
        X = vec_fields.T

        f = np.zeros((2 * self.V, X.shape[1]))
        for v in range(self.V):
            f[v * 2 : ( v + 1 ) * 2] = self.local_bases[v].T @ X[v * 3 : ( v + 1 ) * 3]

        def SampleCovariance(X):
            X_mean = np.mean(X, axis=1)
            X_centered = X - X_mean.reshape(-1,1)
            S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
            return S

        covariance = SampleCovariance(f)

        return CochainSample(
            X=f, 
            covariance=covariance, 
            X_GT=f
            )