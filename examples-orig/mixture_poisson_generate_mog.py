#!/usr/bin/env python
# Generate latent variables from an HVM fit to Mixture
# Poisson model and plot matrix of fixed size.
# KL: TODO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from blackbox.models import PosteriorMixturePoisson
from blackbox.likelihoods import MFPoisson
from blackbox.priors import MixtureGaussians
from blackbox.hvm import HVM
from blackbox.util import discrete_density

if __name__ == '__main__':
  np.random.seed(42)
  model = PosteriorMixturePoisson(np.zeros((2,2)), np.zeros(2))
  q_mf = MFPoisson(model.num_vars)
  q_num_components = 4
  q_prior = MixtureGaussians(q_num_components, q_mf.num_params)
  q_prior.p = np.array(
  [0.1, .1, .1, 0.7]
             )
  q_prior.m = np.array(
  [[1, 0.1, 8, 10], [1, 12, 0.1, 10]]
             ).transpose()
  q_prior.s = np.ones((4,2))*-5

  num_test_samples = 50000
  z_samples = np.zeros((num_test_samples, 2))
  for r in range(num_test_samples):
    lamda_unconst = q_prior.sample()
    lamda = q_mf.transform(lamda_unconst)
    q_mf.set_lamda(lamda)
    z_samples[r, :] = q_mf.sample()

  prob_table = discrete_density(z_samples, support=[[0,20], [0,20]])
  ax = sns.heatmap(prob_table,
                   linewidths=.5, cbar=False, xticklabels=10,
                   yticklabels=False,
                   cmap=sns.cubehelix_palette(8, start=.5, rot=-.75,
                   as_cmap=True), square=True)
  cax = plt.gcf().axes[-1]
  cax.tick_params(labelsize=50)
  ax.xaxis.tick_top()
  sns.plt.show()
