#!/usr/bin/env python
"""
A simple example from Stan. The model is written in NumPy/SciPy.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
import blackbox as bb
import numpy as np

from blackbox import PythonModel
from scipy.stats import beta, bernoulli

class BetaBernoulli(PythonModel):
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def __init__(self):
        self.num_vars = 1

    def _py_log_prob(self, xs, zs):
        # This example is written for pedagogy. We recommend
        # vectorizing operations in practice.
        n_minibatch = zs.shape[0]
        lp = np.zeros(n_minibatch, dtype=np.float32)
        for b in range(n_minibatch):
            lp[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
            for n in range(len(xs)):
                lp[b] += bernoulli.logpmf(xs[n], p=zs[b, :])

        return lp

bb.set_seed(42)
model = BetaBernoulli()
variational = bb.MFBeta(model.num_vars)
data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

inference = bb.MFVI(model, variational, data)
inference.run(n_iter=10000)
