#!/usr/bin/env python
# Generate latent variables from Mixture Poisson model and plot
# matrix of fixed size.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import poisson
from blackbox.models import PosteriorMixturePoisson
from blackbox.util import discrete_density

if __name__ == '__main__':
  np.random.seed(42)
  M = np.array([[1, 0.1, 8, 10], [1, 12, 0.1, 10]])
  pi = np.array([0.1, .1, .1, 0.7])
  model = PosteriorMixturePoisson(M, pi)

  num_test_samples = 20000
  z_samples = np.zeros((num_test_samples, 2))
  for r in range(num_test_samples):
    k = np.random.multinomial(1, pi)
    k = int(np.where(k == 1)[0])
    z_samples[r, 0] = poisson.rvs(M[0, k])
    z_samples[r, 1] = poisson.rvs(M[1, k])

  prob_table = discrete_density(z_samples, support=[[0,20], [0,20]])
  ax = sns.heatmap(prob_table,
                   linewidths=.5, cbar=False, xticklabels=10,
                   yticklabels=10,
                   cmap=sns.cubehelix_palette(8, start=.5, rot=-.75,
                   as_cmap=True), square=True)
  cax = plt.gcf().axes[-1]
  cax.tick_params(labelsize=50)
  ax.xaxis.tick_top()
  sns.plt.show()
