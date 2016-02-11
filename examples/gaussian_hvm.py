#!/usr/bin/env python
# Data
#   X: None
# Probability model
#   Prior p(z): 2-component Mixture of Gaussians
#   Likelihood p(x|z): Student-t
# Discrepancy
#   T(xrep) = max(xrep)
#   xrep: Nx1 matrix
# Variational model
#   z-Likelihood q(z): Mean-field Gaussian
#   t-Likelihood q(t|lambda): Mean-field Gaussian
import numpy as np

import blackbox as bb
from blackbox.models import PosteriorGaussian
from blackbox.likelihoods import MFGaussian
from blackbox.priors import Flow
from blackbox.auxiliaries import InverseFlow

if __name__ == '__main__':
  np.random.seed(143479292)

  model = PosteriorGaussian(1)
  q_mf = MFGaussian(model.num_vars)
  q_flow_length = 4
  q_prior = Flow(q_flow_length, q_mf.num_params)
  #r_flow_length = 20
  r_flow_length = 2*q_flow_length
  r_auxiliary = InverseFlow(r_flow_length, q_mf.num_params)

  inference = bb.HVM(model, q_mf, q_prior, r_auxiliary, niter=int(1e4))
  inference.run()
