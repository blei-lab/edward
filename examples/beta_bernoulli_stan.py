#!/usr/bin/env python
"""
A simple example from Stan. The model is written in Stan.

Probability model
    Prior: Beta
    Likelihood: Bernoulli
Variational model
    Likelihood: Mean-field Beta
"""
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
      theta ~ beta(1.0, 1.0);
      for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
"""
bb.set_seed(42)
model = bb.StanModel(model_code=model_code)
# TODO
# model.num_vars no longer exists in StanModel:
# it doesn't compile until after it takes in data
#variational = bb.MFBeta(model.num_vars)
variational = bb.MFBeta(1)
data = dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

inference = bb.MFVI(model, variational, data)
inference.run(n_iter=10000)
