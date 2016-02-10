import numpy as np
import math

from scipy.stats import norm, multivariate_normal
from util import logistic, logistic_deriv, logistic_hess, tanh, tanh_deriv, \
    tanh_hess, h, h_deriv, h_hess

class Flow:
  """
  Normalizing Flow
  q(lambda; theta) = fk(...(f1(lambda; U1, W1, b1)), Uk, Wk, bk)
  with parameters k x d matrix U, k x d matrix W, k-vector b

  Arguments
  ----------
  flow_length: k
  num_vars: number of latent variables
  """
  def __init__(self, flow_length, num_vars):
    self.flow_length = flow_length
    self.num_vars = num_vars
    self.u = norm.rvs(0, 1, (flow_length, num_vars))
    self.w = norm.rvs(0, 1, (flow_length, num_vars))
    self.b = norm.rvs(0, 1, flow_length)

    c = 1e-4
    self.u_grad = np.zeros((flow_length, num_vars))
    self.w_grad = np.zeros((flow_length, num_vars))
    self.b_grad = np.zeros(flow_length)
    self.u_grad_sum = c * np.ones((flow_length, num_vars))
    self.w_grad_sum = c * np.ones((flow_length, num_vars))
    self.b_grad_sum = c * np.ones(flow_length)
    self.u_momentum = np.zeros((flow_length, num_vars))
    self.w_momentum = np.zeros((flow_length, num_vars))
    self.b_momentum = np.zeros((flow_length))
    self.update_count = 0

    self.lamda_full_sample = np.zeros((flow_length + 1, num_vars))

  def print_params(self):
    print "U:"
    print self.u
    print
    print "W:"
    print self.w
    print
    print "b:"
    print self.b

  def sample(self):
    sample = norm.rvs(0, 1, self.num_vars)
    self.lamda_full_sample[0, :] = sample
    for l in range(self.flow_length):
      sample = self._transform(sample, l)
      self.lamda_full_sample[l + 1, :] = sample

    return sample

  def log_prob(self, sample):
    # TODO
    # right now it doesn't use sample
    # it requires storing self.lamda_full_sample, done sequentially
    # after sampling. generalize this but so that the HVM code still
    # allows generic priors during inference
    sample = self.lamda_full_sample[0, :]
    ret = -math.log(2 * 3.1415) - .5 * np.sum(np.dot(sample, sample))
    for l in range(self.flow_length):
      sample = self.lamda_full_sample[l, :]
      u = self._real_u(l)
      ret -= math.log(1 + np.dot(u, self._psi_helper(sample, l)))
    return ret

  # TODO: Fix the entropy
  # TODO same comment as above
  def add_grad(self, sample, vec_grad):
    """
    updates internally with
    grad_{theta} L(theta, phi) =
      lambda(epsilon; theta) * (vec_grad +
      grad_{lambda} log q(lambda; theta))
    """
    for l in reversed(range(self.flow_length)):
      u = self._real_u(l)
      w = self.w[l, :]
      sample = self.lamda_full_sample[l, :]

      # Get Entropy Grad for log q
      psi_l = self._psi_helper(sample, l)
      self.u_grad[l, :] += psi_l/ (1 + np.dot(u, psi_l))

      # W and beta
      self.w_grad[l, :] += (u * h_deriv(np.dot(w, sample) + self.b[l]) + np.dot(u, w) * (
          h_hess(np.dot(w, sample) + self.b[l]) * sample)) / ( 1 + np.dot(u, psi_l))
      self.b_grad[l] += np.dot(u, w) * h_hess(np.dot(w, sample) + self.b[l]) / ( 1 + np.dot(u, psi_l))

      # Do chain rule for layer l params
      self.u_grad[l, :] += vec_grad * h(np.dot(w, sample) + self.b[l])
      self.w_grad[l, :] += np.dot(vec_grad, u) * h_deriv(np.dot(w, sample) + self.b[l]) * sample
      self.b_grad[l] += np.dot(vec_grad, u) * h_deriv(np.dot(w, sample) + self.b[l])

      # Do chain rule for previous layer
      vec_grad = np.dot(vec_grad, u) * h_deriv(np.dot(w, sample) + self.b[l]) * w + vec_grad

      # Add the z portion of the entropy grad
      vec_grad += np.dot(u, w) * h_hess(np.dot(w, sample) + self.b[l]) * w / ( 1 + np.dot(u, psi_l))

  def normalize_grad(self, normalization):
    self.u_grad /= normalization
    self.w_grad /= normalization
    self.b_grad /= normalization

  def update(self, eta):
#    print self.u_grad
    self._u_chain_rule()

    self.u_grad_sum = (1 - .1) * self.u_grad_sum + .1 * self.u_grad * self.u_grad
    self.w_grad_sum = (1 - .1) * self.w_grad_sum + .1 * self.w_grad * self.w_grad
    self.b_grad_sum = (1 - .1) * self.b_grad_sum + .1 * self.b_grad * self.b_grad

    alpha = 0.9
    self.u_momentum = alpha * self.u_momentum + eta * self.u_grad / np.sqrt(self.u_grad_sum)
    self.w_momentum = alpha * self.w_momentum + eta * self.w_grad / np.sqrt(self.w_grad_sum)
    self.b_momentum = alpha * self.b_momentum + eta * self.b_grad / np.sqrt(self.b_grad_sum)

    self.u += self.u_momentum
    self.w += self.w_momentum
    self.b += self.b_momentum
#    print "U: Rajesh: ", self.u, self.u_grad_sum
#    if self.update_count % 10 == 0:
#      print "W: Rajesh: ", self.w, self.w_grad_sum
#    print "B: Rajesh: ", self.b, self.b_grad_sum

    self.u_grad *= 0
    self.w_grad *= 0
    self.b_grad *= 0
    self.update_count += 1

  def _transform(self, sample, l):
    w = self.w[l, :]
    u_invert = self._real_u(l)
    return sample + u_invert * h(np.dot(w, sample) + self.b[l])

  def _psi_helper(self, sample, l):
    w = self.w[l, :]
    return h_deriv(np.dot(w, sample) + self.b[l]) * w

  def _u_chain_rule(self):
 #   return
    for l in range(self.flow_length):
      w = self.w[l, :]
      u = self.u[l, :]
      w_2 = np.dot(w, w)
      wu_ip = np.dot(w, u)

      u_grad_temp = self.u_grad[l, :] + w * (logistic(np.dot(w, u)) - 1) * np.dot(self.u_grad[l, :], w) / w_2
      w_grad_temp = u * (logistic(np.dot(w, u)) - 1) * np.dot(self.u_grad[l, :], w) / w_2
      w_grad_temp += (-1 + math.log(1 + math.exp(np.dot(w, u)) - np.dot(w, u))) * (
        self.u_grad[l, :] / w_2 - 2 * w * np.dot(w, self.u_grad[l, :]) / w_2 / w_2)

      self.u_grad[l, :] = u_grad_temp
      self.w_grad[l, :] += w_grad_temp

  def _real_u(self, l):
    w = self.w[l, :]
    u = self.u[l, :]
    wu_ip = np.dot(w, u)
    u_invert = u + (-1 + math.log(1 + math.exp(wu_ip)) - wu_ip) * w / np.dot(w, w)
  #  return u
    return u_invert

class MixtureGaussians:
  """
  Mixture of Gaussians
  q(lambda; theta) = \sum_{k=1}^K pi_k N(lambda; mu_k, sigma_k)

  Arguments
  ----------
  num_components: K
  num_vars: number of latent variables
  """
  def __init__(self, num_components, num_vars):
    self.num_components = num_components
    self.num_vars = num_vars
    self.p = norm.rvs(0, 1, num_components)
    self.m = norm.rvs(0, 1, (num_components, num_vars))
    self.s = norm.rvs(0, 1, (num_components, num_vars))

    c = 1e-4
    self.p_grad = np.zeros(num_components)
    self.m_grad = np.zeros((num_components, num_vars))
    self.s_grad = np.zeros((num_components, num_vars))
    self.p_grad_sum = c * np.ones(num_components)
    self.m_grad_sum = c * np.ones((num_components, num_vars))
    self.s_grad_sum = c * np.ones((num_components, num_vars))
    self.p_momentum = np.zeros((num_components))
    self.m_momentum = np.zeros((num_components, num_vars))
    self.s_momentum = np.zeros((num_components, num_vars))

  def print_params(self):
    print "unnormalized pi:"
    print self.p
    print
    print "mu:"
    print self.m
    print
    print "log std sigma:"
    print self.s

  def sample(self):
    k = np.random.multinomial(1, self.p)
    k = int(np.where(k == 1)[0])
    eps = multivariate_normal.rvs(np.zeros(self.num_vars),
                                  np.ones(self.num_vars))
    return self._transform(eps, k)

  def log_prob(self, sample):
    ret = 0
    for k in range(self.num_components):
      ret += self.p[k] * multivariate_normal.pdf(sample, self.m[k, :],
        np.exp(self.s[k, :]))
    return ret

  def add_grad(self, sample, vec_grad):
    """
    updates internally with
    grad_{theta} L(theta, phi) =
      lambda(epsilon; theta) * (vec_grad +
      grad_{lambda} log q(lambda; theta))
    """
    # grad of the transformation
#https://github.com/HIPS/autograd/blob/b935d34d7903e48a994d184710d05355501bc33a/autograd/scipy/stats/multivariate_normal.py
    # grad_lambda
    norm_pdf = np.zeros(self.num_components)
    mog_pdf = 0
    for k in range(self.num_components):
      norm_pdf[k] = multivariate_normal.pdf(sample, self.m[k, :],
        np.exp(self.s[k, :]))
      mog_pdf += self.p[k] * norm_pdf[k]

    for k in range(self.num_components):
      self.p_grad[k] += norm_pdf[k] / mog_pdf
      self.m_grad[k, :] += (self.p[k] *
        _grad_m_multivariate_normal(sample, k, norm_pdf))/ mog_pdf
      self.s_grad[k, :] += (self.p[k] *
        _grad_s_multivariate_normal(sample, k, norm_pdf))/ mog_pdf
      # TODO vec_grad

  def normalize_grad(self, normalization):
    self.p_grad /= normalization
    self.m_grad /= normalization
    self.s_grad /= normalization

  def update(self, eta):
    self.p_grad_sum = (1 - .1) * self.p_grad_sum + .1 * self.p_grad * self.p_grad
    self.m_grad_sum = (1 - .1) * self.m_grad_sum + .1 * self.m_grad * self.m_grad
    self.s_grad_sum = (1 - .1) * self.s_grad_sum + .1 * self.s_grad * self.s_grad

    alpha = 0.9
    self.p_momentum = alpha * self.p_momentum + eta * self.p_grad / np.sqrt(self.p_grad_sum)
    self.m_momentum = alpha * self.m_momentum + eta * self.m_grad / np.sqrt(self.m_grad_sum)
    self.s_momentum = alpha * self.s_momentum + eta * self.s_grad / np.sqrt(self.s_grad_sum)

    self.p += self.p_momentum
    self.m += self.m_momentum
    self.s += self.s_momentum

    self.p_grad *= 0
    self.m_grad *= 0
    self.s_grad *= 0

  def _transform(self, eps, k):
    return self.m[k,:] + np.exp(self.s[k,:]) * eps

  def _grad_m_multivariate_normal(self, sample, k, norm_pdf):
    locs = self.m[k, :]
    scales = np.exp(self.s[k, :])
    return norm_pdf[k] * (sample - locs)/(scales**2)

  def _grad_s_multivariate_normal(self, sample, k, norm_pdf):
    out = np.ones(self.num_vars) * norm_pdf[k]
    for d in range(self.num_vars):
      loc = self.m[k, :]
      scale = np.exp(self.s[k, :])
      out[d] *= -1.0/scale + (sample[d] - loc)**2/scale**3
    return out
