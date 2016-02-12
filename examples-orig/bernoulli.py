#!/usr/bin/env python
# Probability model
#   Posterior: 2 correlated Bernoullis
# Variational model
#   Likelihood: Mean-field Bernoulli
#   Prior: Normalizing flow
#   Auxiliary: Normalizing flow
import numpy as np

import blackbox as bb
from blackbox.models import PosteriorBernoulli
from blackbox.likelihoods import MFBernoulli
from blackbox.priors import Flow
from blackbox.auxiliaries import InverseFlow

if __name__ == '__main__':
  np.random.seed(143479292)
  p = np.array(
    [[0.1, 0.4],
     [0.4, 0.1]])
  model = PosteriorBernoulli(p)

  q_mf = MFBernoulli(model.num_vars)
  q_flow_length = 8
  q_prior = Flow(q_flow_length, q_mf.num_params)
  #r_flow_length = 20
  r_flow_length = 3*q_flow_length
  r_auxiliary = InverseFlow(r_flow_length, q_mf.num_params)

  inference = bb.HVM(model, q_mf, q_prior, r_auxiliary, niter=int(1e4))
  inference.run()
