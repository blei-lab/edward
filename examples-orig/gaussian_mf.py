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

  inference = bb.MFVI(model, q_mf, niter=int(1e3))
  inference.run()
  print inference.q_mf.mu, inference.q_mf.log_std
