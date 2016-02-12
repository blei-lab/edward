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

if __name__ == '__main__':
  np.random.seed(143479292)

  model = PosteriorGaussian(1)
  q_mf = MFGaussian(model.num_vars)
  q_mf.mu = np.array([3.0])
  q_mf.log_std = np.array([3.0])

  inference = bb.AlphaVI(1.0, model, q_mf, niter=int(1e3))
  print inference.q_mf.mu, inference.q_mf.log_std
  inference.run()
  print inference.q_mf.mu, inference.q_mf.log_std
