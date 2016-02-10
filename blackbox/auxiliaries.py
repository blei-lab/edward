import numpy as np
import math

from scipy.stats import norm
from util import logistic, logistic_deriv, logistic_hess, tanh, tanh_deriv, \
    tanh_hess, h, h_deriv, h_hess

class InverseFlow:
  def __init__(self, flow_length, num_vars):
    self.flow_length = flow_length
    self.num_vars = num_vars
    self.u = norm.rvs(0, 1, (flow_length, num_vars))
    self.w = norm.rvs(0, 1, (flow_length, num_vars))
    self.b = norm.rvs(0, 1, flow_length)
    self.w_z = norm.rvs(0, 1, num_vars)

    c = 1e-4
    self.u_grad_sum = c * np.ones((flow_length, num_vars))
    self.w_grad_sum = c * np.ones((flow_length, num_vars))
    self.b_grad_sum = c * np.ones(flow_length)
    self.w_z_grad_sum = c * np.ones(num_vars)

    self.u_momentum = np.zeros((flow_length, num_vars))
    self.w_momentum = np.zeros((flow_length, num_vars))
    self.b_momentum = np.zeros(flow_length)
    self.w_z_momentum = np.zeros(num_vars)

    self.u_grad = np.zeros((flow_length, num_vars))
    self.w_grad = np.zeros((flow_length, num_vars))
    self.b_grad = np.zeros(flow_length)
    self.w_z_grad = np.zeros(num_vars)

    self.sigma_0 = 1

    self.reverse_sample = np.zeros((flow_length + 1, num_vars))

  def reverse_samples(self, sample):
    self.reverse_sample[self.flow_length, :] = sample
    for l in reversed(range(self.flow_length)):
      sample = self._transform(sample, l)
      self.reverse_sample[l, :] = sample

  def likelihood(self, ind, val):
    return -.5 / self.sigma_0 * (self.w_z[ind] * val - self.reverse_sample[0, ind]) ** 2

  def log_prob(self, z_sample):
   ret = 0
   ret += -math.log(2 * 3.1415) - .5 / self.sigma_0 * np.sum((self.w_z * z_sample - self.reverse_sample[0, :]) ** 2)
#   softmax_reverse_sample = log(1 + exp(self.reverse_sample[0, :]))
   for l in range(self.flow_length):
     u = self._real_u(l)
     ret += math.log(1 + np.dot(u, self._psi_helper(self.reverse_sample[l+1, :], l)));
   return ret

  def add_grad(self, z_sample):
    """
    updates internally with
    grad_{phi} L(theta, phi) = grad_{phi} log r(lambda | z; phi)
    returns grad_{lambda} log r(lambda | z; phi)
    """
    # Grad with respect to lambda_0 and for w_z
    self.w_z_grad += (-self.w_z * z_sample + self.reverse_sample[0, :]) * z_sample / self.sigma_0
    running_grad = (self.w_z * z_sample - self.reverse_sample[0, :]) / self.sigma_0

    # March forward for gradients
    for l in range(self.flow_length):
      u = self._real_u(l)
      w = self.w[l, :]
      sample = self.reverse_sample[l + 1, :]

      # Do chain rule for layer l parameters
      self.u_grad[l, :] += running_grad * h(np.dot(w, sample) + self.b[l])
      self.w_grad[l, :] += np.dot(running_grad, u) * h_deriv(np.dot(w, sample) + self.b[l]) * sample
      self.b_grad[l] += np.dot(running_grad, u) * h_deriv(np.dot(w, sample) + self.b[l])

      # Get Entropy Grad for log q
      psi_l = self._psi_helper(sample, l)
      self.u_grad[l, :] += psi_l/ (1 + np.dot(u, psi_l))

      # W and beta
      self.w_grad[l, :] += (u * h_deriv(np.dot(w, sample) + self.b[l]) + np.dot(u, w) * (
          h_hess(np.dot(w, sample) + self.b[l]) * sample)) / ( 1 + np.dot(u, psi_l))
      self.b_grad[l] += np.dot(u, w) * h_hess(np.dot(w, sample) + self.b[l]) / ( 1 + np.dot(u, psi_l))

      # Do chain rule on previous add current
      running_grad = np.dot(running_grad, u) * h_deriv(np.dot(w, sample) + self.b[l]) * w + running_grad

      # Add the z portion of the entropy grad
      running_grad += np.dot(u, w) * h_hess(np.dot(w, sample) + self.b[l]) * w / (1 + np.dot(u, psi_l))

    return running_grad

  def normalize_grad(self, normalization):
    self.u_grad /= normalization
    self.w_grad /= normalization
    self.b_grad /= normalization
    self.w_z_grad /= normalization

  def update(self, eta):
    self._u_chain_rule()
    self.u_grad_sum = (1 - .1) * self.u_grad_sum + .1 * self.u_grad * self.u_grad
    self.w_grad_sum = (1 - .1) * self.w_grad_sum + .1 * self.w_grad * self.w_grad
    self.b_grad_sum = (1 - .1) * self.b_grad_sum + .1 * self.b_grad * self.b_grad
    self.w_z_grad_sum = (1 - .1) * self.w_z_grad_sum + .1 * self.w_z_grad * self.w_z_grad

    alpha = 0.9
    self.u_momentum = alpha * self.u_momentum + eta * self.u_grad / np.sqrt(self.u_grad_sum);
    self.w_momentum = alpha * self.w_momentum + eta * self.w_grad / np.sqrt(self.w_grad_sum);
    self.b_momentum = alpha * self.b_momentum + eta * self.b_grad / np.sqrt(self.b_grad_sum);
    self.w_z_momentum = alpha * self.w_z_momentum + eta * self.w_z_grad / np.sqrt(self.w_z_grad_sum);

    self.u += self.u_momentum
    self.w += self.w_momentum
    self.b += self.b_momentum
    self.w_z += self.w_z_momentum

    self.u_grad *= 0
    self.w_grad *= 0
    self.b_grad *= 0
    self.w_z_grad *= 0

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
    return u_invert

  def _transform(self, sample, l):
    w = self.w[l, :]
    u_invert = self._real_u(l)
    return sample + u_invert * h(np.dot(w, sample) + self.b[l])

  def _psi_helper(self, sample, l):
    w = self.w[l, :]
    return h_deriv(np.dot(w, sample) + self.b[l]) * w
