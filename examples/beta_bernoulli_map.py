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
        lp[0] = beta.logpdf(zs[0,:], a=1.0, b=1.0)
        for n in range(len(xs)):
            lp[0] += bernoulli.logpmf(xs[n], p=zs[0,:])

        return lp


bb.set_seed(42)
model = BetaBernoulli()
data = bb.Data(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
variational = bb.PMBernoulli(1)
inference = bb.MAP(model, variational, data)
inference.run(n_iter=100)
