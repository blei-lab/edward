#!/usr/bin/env python
"""
A simple coin flipping example. The model is written in NumPy/SciPy.
Inspired by Stan's toy example.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np

from edward.models import PythonModel, Variational, Beta
from scipy.stats import beta, bernoulli


class BetaBernoulli(PythonModel):
    """p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)"""
    def _py_log_prob(self, xs, zs):
        # This example is written for pedagogy. We recommend
        # vectorizing operations in practice.
        n_samples = zs.shape[0]
        lp = np.zeros(n_samples, dtype=np.float32)
        for b in range(n_samples):
            lp[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
            for n in range(xs['x'].shape[0]):
                lp[b] += bernoulli.logpmf(xs['x'][n], p=zs[b, :])

        return lp


ed.set_seed(42)
model = BetaBernoulli()
variational = Variational()
variational.add(Beta())
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=10000)
