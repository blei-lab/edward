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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import pymc3 as pm
import theano

from edward.models import PyMC3Model, Variational, Beta

x_obs = theano.shared(np.zeros(1))
with pm.Model() as pm_model:
    beta = pm.Beta('beta', 1, 1, transform=None)
    x = pm.Bernoulli('x', beta, observed=x_obs)

ed.set_seed(42)
model = PyMC3Model(pm_model)
variational = Variational()
variational.add(Beta())
data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=10000)
