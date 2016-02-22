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

from blackbox.util import PythonModel
from scipy.stats import beta, bernoulli

class BetaBernoulli(PythonModel):
    """
    p(z) = Beta(z; 1, 1)
    p(x|z) = Bernoulli(x; z)
    """
    def __init__(self, data):
        self.data = data
        self.num_vars = 1

    def _py_log_prob(self, zs):
        # This is written for pedagogy. We recommend vectorizing
        # operations in practice.
        n_minibatch = zs.shape[0]
        log_prob = np.zeros(n_minibatch, dtype=np.float32)
        for b in range(n_minibatch):
            log_prob[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
            for n in range(len(data)):
                log_prob[b] += bernoulli.logpmf(self.data[n], p=zs[b, :])

        return log_prob

bb.set_seed(42)

data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
model = BetaBernoulli(data)
q = bb.MFBeta(model.num_vars)

inference = bb.MFVI(model, q, n_minibatch=5)
inference.run()
