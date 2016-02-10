#!/usr/bin/env python
# Probability model
#   Posterior: Mixture Poisson
# Variational model
#   Likelihood: Mean-field Poisson
#   Prior: Mixture of Gaussians
#   Auxiliary: Normalizing flow
import numpy as np

from hvm.models import PosteriorMixturePoisson
from hvm.likelihoods import MFPoisson
from hvm.priors import MixtureGaussians
from hvm.auxiliaries import InverseFlow
from hvm.hvm import HVM

if __name__ == '__main__':
  np.random.seed(42)
  M = np.array([[1, 0.1, 5, 10], [1, 12, 0.01, 10]])
  pi = np.array([0.07, .09, .04, 0.7])
  model = PosteriorMixturePoisson(M, pi)

  q_mf = MFPoisson(model.num_vars)
  q_num_components = 4
  q_prior = MixtureGaussians(q_num_components, q_mf.num_params)
  q_prior.p = np.array(
  [0.07, .09, .04, 0.7]
             )
  q_prior.m = np.array(
  [[1, 0.1, 5, 10], [1, 12, 0.01, 10]]
             ).transpose()
  q_prior.s = np.ones((4,2))*-5
  r_flow_length = 20
  r_auxiliary = InverseFlow(r_flow_length, q_mf.num_params)

  vi = HVM(model, q_mf, q_prior, r_auxiliary, niter=int(1e4))
  vi.run()
