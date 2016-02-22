#!/usr/bin/env python
"""
A simple example from Stan.
Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
import tensorflow as tf
import blackbox as bb

model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(0.5, 0.5);  // Jeffreys' prior
      for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
"""
data = dict(N=10, y=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1])

bb.set_seed(42)
model = bb.Model(model_code=model_code, data=data)
q = bb.MFBeta(model.num_vars)

inference = bb.MFVI(model, q, n_minibatch=1)
inference.run()
