#!/usr/bin/env python
# Probability model
#   Posterior: 2 correlated Bernoullis
# Variational model
#   Likelihood: Mean-field Bernoulli
#   Prior: Normalizing flow
#   Auxiliary: Normalizing flow
import numpy as np

from hvm.models import PosteriorBernoulli
from hvm.likelihoods import MFBernoulli
from hvm.priors import Flow
from hvm.auxiliaries import InverseFlow
from hvm.hvm import HVM

if __name__ == '__main__':
  np.random.seed(143479292)
  p = np.array(
    [[[0.03, 0.06],
      [0.06, 0.12]],
     [[0.06, 0.12],
      [0.12, 0.43]]])
  model = PosteriorBernoulli(p)

  q_mf = MFBernoulli(model.num_vars)
  q_flow_length = 8
  q_prior = Flow(q_flow_length, q_mf.num_params)
  r_flow_length = 24
  r_auxiliary = InverseFlow(r_flow_length, q_mf.num_params)

  vi = HVM(model, q_mf, q_prior, r_auxiliary, niter=int(1e4))
  vi.run()
