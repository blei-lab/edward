#!/usr/bin/env python
"""
A simple coin flipping example. The model is written in PyMC3.
Inspired by Stan's toy example.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
import edward as ed
import pymc3 as pm
import numpy as np
import theano

from edward.models import PyMC3Model, Variational, Beta

data_shared = theano.shared(np.zeros(1))

with pm.Model() as model:
    beta = pm.Beta('beta', 1, 1, transform=None)
    out = pm.Bernoulli('data',
                       beta,
                       observed=data_shared)

data = {'TODO': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
m = PyMC3Model(model, data_shared)
variational = Variational()
variational.add(Beta())

inference = ed.MFVI(m, variational, data)
inference.run(n_iter=10000)
