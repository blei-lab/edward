#!/usr/bin/env python
# Generate latent variables from a mean-field fit to Mixture
# Poisson model and plot matrix of fixed size.
# KL: TODO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from blackbox.likelihoods import MFPoisson
from blackbox.hvm import HVM
from blackbox.util import discrete_density

if __name__ == '__main__':
  np.random.seed(42)
  lamda = np.array([ 10.00029369, 10.02984577])
  q_mf = MFPoisson(2)
  q_mf.set_lamda(lamda)

  num_test_samples = 5000
  z_samples = np.zeros((num_test_samples, 2))
  for r in range(num_test_samples):
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
