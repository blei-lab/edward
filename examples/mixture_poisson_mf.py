#!/usr/bin/env python
# Probability model
#   Posterior: Mixture Poisson
# Variational model
#   Likelihood: Mean-field Poisson
import numpy as np

from blackbox.models import PosteriorMixturePoisson
from blackbox.likelihoods import MFPoisson
from blackbox.hvm import MFVI

if __name__ == '__main__':
  np.random.seed(42)
  M = np.array([[1, 0.1, 5, 10], [1, 12, 0.01, 10]])
  pi = np.array([0.07, .09, .04, 0.7])
  model = PosteriorMixturePoisson(M, pi)

  q_mf = MFPoisson(model.num_vars)

  vi = MFVI(model, q_mf, niter=int(1e4))
  vi.run()
