#!/usr/bin/env python
"""
The model is written in NumPy/SciPy.

Probability model
    Posterior: (1-dimensional) Gaussian
Variational model
    Likelihood: Mean-field Gaussian
"""
import blackbox as bb
import numpy as np

from blackbox.util import PythonModel
from scipy.stats import norm

class Gaussian(PythonModel):
    """
    p(x, z) = p(z) = p(z | x) = Gaussian(z; mu, Sigma)
    """
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma
        self.num_vars = 1

    def _py_log_prob(self, zs):
        # This example is written for pedagogy. We recommend
        # vectorizing operations in practice.
        n_minibatch = zs.shape[0]
        log_prob = np.zeros(n_minibatch, dtype=np.float32)
        for b in range(n_minibatch):
            log_prob[b] = norm.logpdf(zs[b, :], loc=self.mu, b=self.Sigma)

        return log_prob

    def _py_log_prob_grad(self, zs):
        return np.array([0, 0], dtype=np.float32)


bb.set_seed(42)

mu = np.array([1.0])
Sigma = np.array([1.0])
model = Gaussian(mu, Sigma)
q = bb.MFGaussian(model.num_vars)

inference = bb.MFVI(model, q, n_iter=10000)
inference.run()
