"""Mixture of Gaussians, with block Gibbs for inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import (
    Dirichlet, Categorical, InverseGamma, ParamMixture, Normal)


def main(_):
  # Generate data
  true_mu = np.array([-1.0, 0.0, 1.0], np.float32) * 10
  true_sigmasq = np.array([1.0**2, 2.0**2, 3.0**2], np.float32)
  true_pi = np.array([0.2, 0.3, 0.5], np.float32)
  N = 10000
  K = len(true_mu)
  true_z = np.random.choice(np.arange(K), size=N, p=true_pi)
  x_data = true_mu[true_z] + np.random.randn(N) * np.sqrt(true_sigmasq[true_z])

  # Prior hyperparameters
  pi_alpha = np.ones(K, dtype=np.float32)
  mu_sigma = np.std(true_mu)
  sigmasq_alpha = 1.0
  sigmasq_beta = 2.0

  # Model
  pi = Dirichlet(pi_alpha)
  mu = Normal(0.0, mu_sigma, sample_shape=K)
  sigmasq = InverseGamma(sigmasq_alpha, sigmasq_beta, sample_shape=K)
  x = ParamMixture(pi, {'loc': mu, 'scale': tf.sqrt(sigmasq)}, Normal,
                   sample_shape=N)
  z = x.cat

  # Conditionals
  mu_cond = ed.complete_conditional(mu)
  sigmasq_cond = ed.complete_conditional(sigmasq)
  pi_cond = ed.complete_conditional(pi)
  z_cond = ed.complete_conditional(z)

  sess = ed.get_session()

  # Initialize randomly
  pi_est, mu_est, sigmasq_est, z_est = sess.run([pi, mu, sigmasq, z])

  print('Initial parameters:')
  print('pi:', pi_est)
  print('mu:', mu_est)
  print('sigmasq:', sigmasq_est)
  print()

  # Gibbs sampler
  cond_dict = {pi: pi_est, mu: mu_est, sigmasq: sigmasq_est,
               z: z_est, x: x_data}
  t0 = time()
  T = 500
  for t in range(T):
    z_est = sess.run(z_cond, cond_dict)
    cond_dict[z] = z_est
    pi_est, mu_est = sess.run([pi_cond, mu_cond], cond_dict)
    cond_dict[pi] = pi_est
    cond_dict[mu] = mu_est
    sigmasq_est = sess.run(sigmasq_cond, cond_dict)
    cond_dict[sigmasq] = sigmasq_est
  print('took %.3f seconds to run %d iterations' % (time() - t0, T))

  print()
  print('Final sample for parameters::')
  print('pi:', pi_est)
  print('mu:', mu_est)
  print('sigmasq:', sigmasq_est)
  print()

  print()
  print('True parameters:')
  print('pi:', true_pi)
  print('mu:', true_mu)
  print('sigmasq:', true_sigmasq)
  print()

  plt.figure(figsize=[10, 10])
  plt.subplot(2, 1, 1)
  plt.hist(x_data, 50)
  plt.title('Empirical Distribution of $x$')
  plt.xlabel('$x$')
  plt.ylabel('frequency')
  xl = plt.xlim()
  plt.subplot(2, 1, 2)
  plt.hist(sess.run(x, {pi: pi_est, mu: mu_est, sigmasq: sigmasq_est}), 50)
  plt.title("Predictive distribution $p(x \mid \mathrm{inferred }\ "
            "\pi, \mu, \sigma^2)$")
  plt.xlabel('$x$')
  plt.ylabel('frequency')
  plt.xlim(xl)
  plt.show()

if __name__ == "__main__":
  plt.style.use('ggplot')
  tf.app.run()
