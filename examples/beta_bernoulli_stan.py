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
data = dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

bb.set_seed(42)
model = bb.StanModel(model_code=model_code, data=data)
q = bb.MFBeta(model.num_vars)

inference = bb.MFVI(model, q)
inference.run(n_iter=10000)
