#!/usr/bin/env python
# Generate latent variables from Exponential Family with Gaussian
# prior on natural parameters and plot
# matrix of fixed size.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import multivariate_normal, gamma
from hvm.util import discrete_density

if __name__ == '__main__':
  np.random.seed(42)
  K = 5 # multinomial number
  m = np.array([[0.07, .09, .04, 0.7]])
  s = np.array([0.07, .09, .04, 0.7])

  num_test_samples = 50000
  lamda_unconst_samples = np.zeros((num_test_samples, 2))
  lamda_samples = np.zeros((num_test_samples, 2))
  z_samples = np.zeros((num_test_samples, 2))
  mu = np.zeros(2)-2
  Sigma = np.array([[1, -.90], [-.90, 1]])*1
  for n in range(num_test_samples):
    lamda_unconst_samples[n, :] = \
      multivariate_normal.rvs(mu, Sigma)
    for d in range(2):
      lamda_samples[n, d] = np.log(1 + np.exp(lamda_unconst_samples[n, d]))
    z_samples[n, 0] = gamma.rvs(lamda_samples[n, 0], size=1) / 100
    z_samples[n, 1] = gamma.rvs(lamda_samples[n, 1], size=1) / 100

  y = np.zeros(num_test_samples)
  for n in range(num_test_samples):
    y[n] = multivariate_normal.pdf(lamda_unconst_samples[n, :],
      mu, Sigma)

  sns.set_style("white")
  sns.mpl.rc("figure", figsize=(9,9))
  ax = sns.kdeplot(lamda_unconst_samples, y,
                   shade=True, cmap="Reds", square=True).set( \
                   xlim=(-6, 2), ylim=(-6, 2),
                   xticks=[-6,2],
                   yticks=[-6,2])
  cax = plt.gcf().axes[-1]
  cax.tick_params(labelsize=50)
  #sns.plt.show()
  plt.savefig("gaussian_1.png")

  # i use photoshop to copy and paste this with the above one.
  sns.set_style("white")
  ax = sns.kdeplot(lamda_unconst_samples, y,
                   shade=True, cmap="Blues").set( \
                   xlim=(-6, 2), ylim=(-6, 2),
                   xticks=[-6,2],
                   yticks=[-6,2])
  cax = plt.gcf().axes[-1]
  cax.tick_params(labelsize=50)
  plt.savefig("gaussian_2.png")
  #sns.plt.show()

  idx = lamda_unconst_samples[:, 0] < mu[0]

  sns.set_style("white")
  #ax = sns.kdeplot(lamda_samples[idx, 0],
  #                 lamda_samples[idx, 1],
  ax = sns.kdeplot(lamda_samples, lamda_samples,
                   shade=True, cmap="Blues", shade_lowest=False).set( \
                   xlim=(0, 1), ylim=(0, 1),
                   xticks=[0,1],
                   yticks=[0,1])
  cax = plt.gcf().axes[-1]
  cax.tick_params(labelsize=50)
  plt.savefig("gamma_1.png")
  #sns.plt.show()

  sns.set_style("white")
  #ax = sns.kdeplot(lamda_samples[np.logical_not(idx), 0],
  #                 lamda_samples[np.logical_not(idx), 1],
  ax = sns.kdeplot(lamda_samples, lamda_samples,
                   shade=True, cmap="Reds", shade_lowest=False).set( \
                   xlim=(0, 1), ylim=(0, 1),
                   xticks=[0,1],
                   yticks=[0,1])
  cax = plt.gcf().axes[-1]
  cax.tick_params(labelsize=50)
  plt.savefig("gamma_2.png")
  #sns.plt.show()
