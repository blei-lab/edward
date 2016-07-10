#!/usr/bin/env python
"""
The model is written in NumPy/SciPy.

Probability model
    Posterior: (1-dimensional) Normal
Variational model
    Likelihood: Mean-field Normal
"""
import autograd.numpy as np
import edward as ed

from autograd.scipy.stats import norm
from edward.models import PythonModel, Variational, Normal

class NormalPosterior(PythonModel):
    """
    p(x, z) = p(z) = p(z | x) = Gaussian(z; mu, Sigma)
    """
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def _py_log_prob(self, xs, zs):
        # This example is written for pedagogy. We recommend
        # vectorizing operations in practice.
        n_samples = zs.shape[0]
        lp = np.zeros(n_samples, dtype=np.float32)
        for b in range(n_samples):
            lp[b] = norm.logpdf(zs[b, :], loc=self.mu, scale=self.std)

        return lp

ed.set_seed(42)
mu = np.array(1.0)
std = np.array(1.0)
model = NormalPosterior(mu, std)
variational = Variational()
variational.add(Normal())

inference = ed.MFVI(model, variational)
inference.run(n_iter=10000)
