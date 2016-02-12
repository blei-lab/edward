#!/usr/bin/env python
# Probability model
#   Posterior: Mixture Poisson
# Variational model
#   Likelihood: Mean-field Poisson
#   Prior: Normalizing flow
#   Auxiliary: Normalizing flow
import numpy as np

import blackbox as bb
from blackbox.models import PosteriorMixturePoisson
from blackbox.likelihoods import MFPoisson
from blackbox.priors import Flow
from blackbox.auxiliaries import InverseFlow

if __name__ == '__main__':
  np.random.seed(42)
  M = np.array([[1, 0.1, 5, 10], [1, 12, 0.01, 10]])
  pi = np.array([0.07, .09, .04, 0.7])
  model = PosteriorMixturePoisson(M, pi)

  q_mf = MFPoisson(model.num_vars)
  q_flow_length = 15
  q_prior = Flow(q_flow_length, q_mf.num_params)
  r_flow_length = 30
  r_auxiliary = InverseFlow(r_flow_length, q_mf.num_params)

  inference = bb.HVM(model, q_mf, q_prior, r_auxiliary, niter=int(1e4))
  inference.run()
