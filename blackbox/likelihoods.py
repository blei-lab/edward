import numpy as np
import math
from scipy.stats import bernoulli, norm, poisson

class MFBernoulli:
  """
  q(z | lambda ) = prod_{i=1}^d Bernoulli(z[i] | lambda[i])
  """
  def __init__(self, num_vars):
    self.num_vars = num_vars
    self.num_params = num_vars
    self.lamda = np.zeros(num_vars)

  def sample(self):
    """z ~ q(z | lambda)"""
    z = np.zeros(self.num_vars)
    for d in range(self.num_vars):
      z[d] = bernoulli.rvs(self.lamda[d])

    return z

  def log_prob_zi(self, i, z):
    """log q(z_i | lambda_i)"""
    if i < self.num_vars:
      return bernoulli.logpmf(z[i], self.lamda[i])
    else:
      raise

  def score_zi(self, i, z):
    """ nabla_{lambda_i} log q(z_i | lambda_i)"""
    if i < self.num_params:
      return z[i] - self.lamda[i]
    else:
      raise

  def transform(self, lamda_unconst):
    """Transform values to be in supp(lambda), q(z | lambda)."""
    lamda = np.zeros(self.num_vars)
    for d in range(self.num_vars):
      lamda[d] = logistic(lamda_unconst[d])

    lamda = self.truncate(lamda)
    return lamda

  def truncate(self, lamda):
    """Truncate values to be in supp(lambda), q(z | lambda)."""
    for d in range(self.num_vars):
      lamda[d] = max(lamda[d], 1e-300)
      lamda[d] = min(lamda[d], .9999999999999)
    return lamda

  def set_lamda(self, lamda):
    self.lamda = lamda

  def add_lamda(self, lamda):
    self.lamda += lamda

class MFGaussian:
  """
  q(z | lambda ) = prod_{i=1}^d Gaussian(z[i] | lambda[i])
  """
  def __init__(self, num_vars):
    self.num_vars = num_vars
    self.num_params = 2*num_vars
    self.mu = norm.rvs(0, 1, num_vars)
    self.log_std = norm.rvs(0, 1, num_vars)

  def sample(self):
    """z ~ q(z | lambda)"""
    z = np.zeros(self.num_vars)
    for d in range(self.num_vars):
      z[d] = norm.rvs(self.mu[d], np.exp(self.log_std[d]))

    return z

  def log_prob_zi(self, i, z):
    """log q(z_i | lambda_i)"""
    if i < self.num_vars:
      return norm.logpdf(z[i], self.mu[i], np.exp(self.log_std[i]))
    else:
      raise

  def score_zi(self, i, z):
    """nabla_{lambda_i} log q(z_i | lambda_i)
    Currently split so that the output is
    (nabla_{mu_1}, nabla_{log_std_1}, ..., nabla_{mu_d},
     nabla_{log_std_d})
    """
    # TODO this is bound to break things in the HVM class
    if i < self.num_vars:
      zi = z[i]
      mui = self.mu[i]
      log_stdi = self.log_std[i]
      return np.array([np.exp(-2*log_stdi)*(zi - mui),
                       np.exp(-2*log_stdi)*(zi - mui)**2 - 1])
    else:
      raise

  def transform(self, lamda_unconst):
    """Transform values to be in supp(lambda), q(z | lambda)."""
    return self.truncate(lamda_unconst)

  def truncate(self, lamda):
    """Truncate values to be in supp(lambda), q(z | lambda)."""
    return lamda

  def set_lamda(self, lamda):
    self.mu = lamda[[2*i for i in range(self.num_vars)]]
    self.log_std = lamda[[2*i+1 for i in range(self.num_vars)]]

  def add_lamda(self, lamda):
    mu = lamda[[2*i for i in range(self.num_vars)]]
    log_std = lamda[[2*i+1 for i in range(self.num_vars)]]
    self.mu += mu
    self.log_std += log_std

class MFPoisson:
  """
  q(z | lambda ) = prod_{i=1}^d Poisson(z[i] | lambda[i])
  """
  def __init__(self, num_vars):
    self.num_vars = num_vars
    self.num_params = num_vars
    self.lamda = np.ones(num_vars)

  def sample(self):
    """z ~ q(z | lambda)"""
    z = np.zeros(self.num_vars)
    for d in range(len(z)):
      z[d] = poisson.rvs(self.lamda[d])

    return z

  def log_prob_zi(self, i, z):
    """log q(z_i | lambda_i)"""
    if i < self.num_vars:
      return poisson.logpmf(z[i], self.lamda[i])
    else:
      raise

  def score_zi(self, i, z):
    """nabla_{lambda_i} log q(z_i | lambda_i)"""
    if i < self.num_params:
      zi = z[i]
      lambdai = self.lamda[i]
      return zi / lambdai - 1
    else:
      raise

  def transform(self, lamda_unconst):
    """Transform values to be in supp(lambda), q(z | lambda)."""
    lamda = np.zeros(self.num_vars)
    for d in range(self.num_vars):
      lamda[d] = np.log(1 + np.exp(lamda_unconst[d]))

    lamda = self.truncate(lamda)
    return lamda

  def truncate(self, lamda):
    """Truncate values to be in supp(lambda), q(z | lambda)."""
    for d in range(self.num_vars):
      lamda[d] = max(lamda[d], 1e-300)
    return lamda

  def set_lamda(self, lamda):
    self.lamda = lamda

  def add_lamda(self, lamda):
    self.lamda += lamda
