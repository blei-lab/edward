import numpy as np

from scipy.stats import poisson, norm
from util import logistic

class PosteriorBernoulli:
  """
  Posterior p(z), where z=(z1,...,zd) ~ Bernoulli(p), with no data

  Arguments
  ----------
  p: d x .... x d table of probabilities
  """
  def __init__(self, p):
    self.lp_ = np.log(p)
    self.num_vars = len(p.shape)

  def log_prob(self, x, z):
    """
    Arguments
    ----------
    x: data (dummy variable)
    z: latent variables (z1, ..., zd) \in {0,1}^d
    """
    elem = self.lp_
    for d in range(self.num_vars):
      elem = elem[z[d]]
    return elem

class PosteriorMixturePoisson:
  """
  Posterior p(z1,z2) with no data, where

  z1 ~ Poisson(\sum_{k=1}^K \pi_k M_{1k})
  z2 ~ Poisson(\sum_{k=1}^K \pi_k M_{2k})

  Arguments
  ----------
  M: 2 x K matrix of positive rates
  pi: K-component vector of membership probabilities
  """
  def __init__(self, M, pi):
    self.M_ = M
    self.pi_ = pi
    self.num_vars = 2

  def log_prob(self, x, z):
    """
    Arguments
    ----------
    x: data (dummy variable)
    z: latent variables (z1, z2) \in {0,1,...}^2
    """
    lp = 0
    for d in range(self.num_vars):
      lp += poisson.logpmf(z[d], np.dot(self.pi_, self.M_[d, :]))
    return lp

class PosteriorGaussian:
  def __init__(self, num_vars):
    self.num_vars = num_vars
    self.mu = np.zeros(num_vars)
    self.log_std = np.zeros(num_vars)

  def log_prob(self, x, z):
    return norm.logpdf(z, self.mu, np.exp(self.log_std))
