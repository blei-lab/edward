from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward import models as rvs


class test_conjugacy_class(tf.test.TestCase):

  def test_basic_bernoulli(self):
    N = 10
    z = rvs.Bernoulli(probs=0.75, sample_shape=N)
    z_cond = ed.complete_conditional(z, [z])
    self.assertIsInstance(z_cond, rvs.Bernoulli)

    with self.test_session() as sess:
      p_val = sess.run(z_cond.probs)

    self.assertAllClose(p_val, 0.75 + np.zeros(N, np.float32))

  def test_incomplete_blanket(self):
    N = 10
    z = rvs.Bernoulli(probs=0.75, sample_shape=N)
    z_cond = ed.complete_conditional(z, [])
    self.assertIsInstance(z_cond, rvs.Bernoulli)

    with self.test_session() as sess:
      p_val = sess.run(z_cond.probs)

    self.assertAllClose(p_val, 0.75 + np.zeros(N, np.float32))

  def test_missing_blanket(self):
    N = 10
    z = rvs.Bernoulli(probs=0.75, sample_shape=N)
    z_cond = ed.complete_conditional(z)
    self.assertIsInstance(z_cond, rvs.Bernoulli)

    with self.test_session() as sess:
      p_val = sess.run(z_cond.probs)

    self.assertAllClose(p_val, 0.75 + np.zeros(N, np.float32))

  def test_blanket_changes(self):
    pi = rvs.Dirichlet(tf.ones(3))
    mu = rvs.Normal(0.0, 1.0)
    z = rvs.Categorical(probs=pi)

    pi1_cond = ed.complete_conditional(pi, [z, pi])
    pi2_cond = ed.complete_conditional(pi, [z, mu, pi])

    self.assertIsInstance(pi1_cond, rvs.Dirichlet)
    self.assertIsInstance(pi2_cond, rvs.Dirichlet)

    with self.test_session() as sess:
      conc1_val, conc2_val = sess.run([pi1_cond.concentration,
                                       pi2_cond.concentration])

    self.assertAllClose(conc1_val, conc2_val)

  def test_beta_bernoulli(self):
    x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

    a0 = 0.5
    b0 = 1.5
    pi = rvs.Beta(a0, b0)
    x = rvs.Bernoulli(probs=pi, sample_shape=10)

    pi_cond = ed.complete_conditional(pi, [pi, x])

    self.assertIsInstance(pi_cond, rvs.Beta)

    with self.test_session() as sess:
      a_val, b_val = sess.run([pi_cond.concentration1,
                               pi_cond.concentration0], {x: x_data})

    self.assertAllClose(a_val, a0 + x_data.sum())
    self.assertAllClose(b_val, b0 + (1 - x_data).sum())

  def test_beta_binomial(self):
    n_data = 10
    x_data = 2

    a0 = 0.5
    b0 = 1.5
    pi = rvs.Beta(a0, b0)
    # use value since cannot sample
    x = rvs.Binomial(total_count=n_data, probs=pi, value=0.0)

    pi_cond = ed.complete_conditional(pi, [pi, x])

    self.assertIsInstance(pi_cond, rvs.Beta)

    with self.test_session() as sess:
      a_val, b_val = sess.run([pi_cond.concentration1,
                               pi_cond.concentration0], {x: x_data})

    self.assertAllClose(a_val, a0 + x_data)
    self.assertAllClose(b_val, b0 + n_data - x_data)

  def test_gamma_exponential(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    alpha0 = 0.5
    beta0 = 1.75
    lam = rvs.Gamma(alpha0, beta0)
    x = rvs.Exponential(lam, sample_shape=4)

    lam_cond = ed.complete_conditional(lam, [lam, x])

    self.assertIsInstance(lam_cond, rvs.Gamma)

    with self.test_session() as sess:
      alpha_val, beta_val = sess.run(
          [lam_cond.concentration, lam_cond.rate], {x: x_data})

    self.assertAllClose(alpha_val, alpha0 + len(x_data))
    self.assertAllClose(beta_val, beta0 + x_data.sum())

  def test_gamma_poisson(self):
    x_data = np.array([0, 1, 0, 7, 0, 0, 2, 0, 0, 1])

    alpha0 = 0.5
    beta0 = 1.75
    lam = rvs.Gamma(alpha0, beta0)
    # use value since cannot sample
    x = rvs.Poisson(lam, value=tf.zeros(10), sample_shape=10)

    lam_cond = ed.complete_conditional(lam, [lam, x])

    self.assertIsInstance(lam_cond, rvs.Gamma)

    with self.test_session() as sess:
      alpha_val, beta_val = sess.run(
          [lam_cond.concentration, lam_cond.rate], {x: x_data})

    self.assertAllClose(alpha_val, alpha0 + x_data.sum())
    self.assertAllClose(beta_val, beta0 + len(x_data))

  def test_gamma_gamma(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    alpha0 = 0.5
    beta0 = 1.75
    alpha_likelihood = 2.3
    beta = rvs.Gamma(alpha0, beta0)
    x = rvs.Gamma(alpha_likelihood, beta, sample_shape=4)

    beta_cond = ed.complete_conditional(beta, [beta, x])

    self.assertIsInstance(beta_cond, rvs.Gamma)

    with self.test_session() as sess:
      alpha_val, beta_val = sess.run(
          [beta_cond.concentration, beta_cond.rate], {x: x_data})
    self.assertAllClose(alpha_val, alpha0 + alpha_likelihood * len(x_data))
    self.assertAllClose(beta_val, beta0 + x_data.sum())

  def test_mul_rate_gamma(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    alpha0 = 0.5
    beta0 = 1.75
    alpha_likelihood = 2.3
    beta = rvs.Gamma(alpha0, beta0)
    x = rvs.Gamma(alpha_likelihood, alpha_likelihood * beta, sample_shape=4)

    beta_cond = ed.complete_conditional(beta, [beta, x])

    self.assertIsInstance(beta_cond, rvs.Gamma)

    with self.test_session() as sess:
      alpha_val, beta_val = sess.run([beta_cond.concentration, beta_cond.rate],
                                     {x: x_data})
    self.assertAllClose(alpha_val, alpha0 + alpha_likelihood * len(x_data))
    self.assertAllClose(beta_val, beta0 + alpha_likelihood * x_data.sum())

  def test_normal_normal(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    mu0 = 0.3
    sigma0 = 2.1
    sigma_likelihood = 1.2

    mu = rvs.Normal(mu0, sigma0)
    x = rvs.Normal(mu, sigma_likelihood, sample_shape=len(x_data))

    mu_cond = ed.complete_conditional(mu, [mu, x])
    self.assertIsInstance(mu_cond, rvs.Normal)

    with self.test_session() as sess:
      mu_val, sigma_val = sess.run([mu_cond.loc, mu_cond.scale], {x: x_data})

    self.assertAllClose(sigma_val, (1.0 / sigma0**2 +
                                    len(x_data) / sigma_likelihood**2) ** -0.5)
    self.assertAllClose(mu_val,
                        sigma_val**2 * (mu0 / sigma0**2 +
                                        (1.0 / sigma_likelihood**2 *
                                         x_data.sum())))

  def test_inverse_gamma_normal(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    sigmasq_conc = 1.3
    sigmasq_rate = 2.1
    x_loc = 0.3

    sigmasq = rvs.InverseGamma(sigmasq_conc, sigmasq_rate)
    x = rvs.Normal(x_loc, tf.sqrt(sigmasq), sample_shape=len(x_data))

    sigmasq_cond = ed.complete_conditional(sigmasq, [sigmasq, x])
    self.assertIsInstance(sigmasq_cond, rvs.InverseGamma)

    with self.test_session() as sess:
      conc_val, rate_val = sess.run(
          [sigmasq_cond.concentration, sigmasq_cond.rate], {x: x_data})

    self.assertAllClose(conc_val, sigmasq_conc + 0.5 * len(x_data))
    self.assertAllClose(rate_val,
                        sigmasq_rate + 0.5 * np.sum((x_data - x_loc)**2))

  def test_normal_normal_scaled(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    mu0 = 0.3
    sigma0 = 2.1
    sigma_likelihood = 1.2
    c = 2.0

    mu = rvs.Normal(mu0, sigma0)
    x = rvs.Normal(c * mu, sigma_likelihood, sample_shape=len(x_data))

    mu_cond = ed.complete_conditional(mu, [mu, x])
    self.assertIsInstance(mu_cond, rvs.Normal)

    with self.test_session() as sess:
      mu_val, sigma_val = sess.run([mu_cond.loc, mu_cond.scale], {x: x_data})

    self.assertAllClose(sigma_val,
                        (1.0 / sigma0**2 +
                         c**2 * len(x_data) / sigma_likelihood**2) ** -0.5)
    self.assertAllClose(mu_val,
                        sigma_val**2 * (mu0 / sigma0**2 +
                                        (c / sigma_likelihood**2 *
                                         x_data.sum())))

  def test_dirichlet_categorical(self):
    x_data = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], np.int32)
    N = x_data.shape[0]
    D = x_data.max() + 1

    alpha = np.zeros(D).astype(np.float32) + 2.0

    theta = rvs.Dirichlet(alpha)
    x = rvs.Categorical(probs=theta, sample_shape=N)

    theta_cond = ed.complete_conditional(theta, [theta, x])

    with self.test_session() as sess:
      alpha_val = sess.run(theta_cond.concentration, {x: x_data})

    self.assertAllClose(alpha_val, np.array([6.0, 5.0, 4.0, 3.0], np.float32))

  def test_dirichlet_multinomial(self):
    x_data = np.array([4, 3, 2, 1], np.int32)
    N = x_data.sum()
    D = x_data.shape[0]

    alpha = np.zeros(D).astype(np.float32) + 2.0

    theta = rvs.Dirichlet(alpha)
    x = rvs.Multinomial(total_count=tf.cast(N, tf.float32), probs=theta)

    theta_cond = ed.complete_conditional(theta, [theta, x])

    with self.test_session() as sess:
      alpha_val = sess.run(theta_cond.concentration, {x: x_data})

    self.assertAllClose(alpha_val, np.array([6.0, 5.0, 4.0, 3.0], np.float32))

  def test_mog(self):
    x_val = np.array([1.1, 1.2, 2.1, 4.4, 5.5, 7.3, 6.8], np.float32)
    z_val = np.array([0, 0, 0, 1, 1, 2, 2], np.int32)
    pi_val = np.array([0.2, 0.3, 0.5], np.float32)
    mu_val = np.array([1.0, 5.0, 7.0], np.float32)

    N = x_val.shape[0]
    K = z_val.max() + 1

    pi_alpha = 1.3 + np.zeros(K, dtype=np.float32)
    mu_sigma = 4.0
    sigmasq = 2.0**2

    pi = rvs.Dirichlet(pi_alpha)
    mu = rvs.Normal(0.0, mu_sigma, sample_shape=[K])

    x = rvs.ParamMixture(pi, {'loc': mu, 'scale': tf.sqrt(sigmasq)},
                         rvs.Normal, sample_shape=N)
    z = x.cat

    mu_cond = ed.complete_conditional(mu)
    pi_cond = ed.complete_conditional(pi)
    z_cond = ed.complete_conditional(z)

    with self.test_session() as sess:
      pi_cond_alpha, mu_cond_mu, mu_cond_sigma, z_cond_p = (
          sess.run([pi_cond.concentration, mu_cond.loc,
                    mu_cond.scale, z_cond.probs],
                   {z: z_val, x: x_val, pi: pi_val, mu: mu_val}))

    true_pi = pi_alpha + np.unique(z_val, return_counts=True)[1]
    self.assertAllClose(pi_cond_alpha, true_pi)
    for k in range(K):
      sigmasq_true = (1.0 / 4**2 + 1.0 / sigmasq * (z_val == k).sum())**-1
      mu_true = sigmasq_true * (1.0 / sigmasq * x_val[z_val == k].sum())
      self.assertAllClose(np.sqrt(sigmasq_true), mu_cond_sigma[k])
      self.assertAllClose(mu_true, mu_cond_mu[k])
    true_log_p_z = np.log(pi_val) - 0.5 / sigmasq * (x_val[:, np.newaxis] -
                                                     mu_val)**2
    true_log_p_z -= true_log_p_z.max(1, keepdims=True)
    true_p_z = np.exp(true_log_p_z)
    true_p_z /= true_p_z.sum(1, keepdims=True)
    self.assertAllClose(z_cond_p, true_p_z)

if __name__ == '__main__':
  tf.test.main()
