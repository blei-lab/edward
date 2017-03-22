from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward import models as rvs
from edward.inferences import conjugacy as conj


class test_conjugacy_class(tf.test.TestCase):

  def test_basic_bernoulli(self):
    N = 10
    z = rvs.Bernoulli(p=0.75, sample_shape=N)
    z_cond = conj.complete_conditional(z, [z])
    self.assertIsInstance(z_cond, rvs.Bernoulli)

    sess = tf.InteractiveSession()
    p_val = sess.run(z_cond.p)

    self.assertAllClose(p_val, 0.75 + np.zeros(N, np.float32))

  def test_beta_bernoulli(self):
    x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

    a0 = 0.5
    b0 = 1.5
    pi = rvs.Beta(a=a0, b=b0)
    x = rvs.Bernoulli(p=pi, sample_shape=10)

    pi_cond = conj.complete_conditional(pi, [pi, x])

    self.assertIsInstance(pi_cond, rvs.Beta)

    sess = tf.InteractiveSession()
    a_val, b_val = sess.run([pi_cond.a, pi_cond.b], {x: x_data})

    self.assertAllClose(a_val, a0 + x_data.sum())
    self.assertAllClose(b_val, b0 + (1-x_data).sum())

  def test_gamma_poisson(self):
    x_data = np.array([0, 1, 0, 7, 0, 0, 2, 0, 0, 1])

    alpha0 = 0.5
    beta0 = 1.75
    lam = rvs.Gamma(alpha=alpha0, beta=beta0)
    x = rvs.Poisson(lam=lam, value=x_data)

    lam_cond = conj.complete_conditional(lam, [lam, x])

    self.assertIsInstance(lam_cond, rvs.Gamma)

    sess = tf.InteractiveSession()
    alpha_val, beta_val = sess.run([lam_cond.alpha, lam_cond.beta], {x: x_data})
    self.assertAllClose(alpha_val, alpha0 + x_data.sum())
    self.assertAllClose(beta_val, beta0 + len(x_data))

  def test_gamma_gamma(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    alpha0 = 0.5
    beta0 = 1.75
    alpha_likelihood = 2.3
    beta = rvs.Gamma(alpha=alpha0, beta=beta0)
    x = rvs.Gamma(alpha=alpha_likelihood, beta=beta,
                  value=x_data)

    beta_cond = conj.complete_conditional(beta, [beta, x])

    self.assertIsInstance(beta_cond, rvs.Gamma)

    sess = tf.InteractiveSession()
    alpha_val, beta_val = sess.run([beta_cond.alpha, beta_cond.beta],
                                   {x: x_data})
    self.assertAllClose(alpha_val, alpha0 + alpha_likelihood * len(x_data))
    self.assertAllClose(beta_val, beta0 + x_data.sum())

  def test_mul_rate_gamma(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    alpha0 = 0.5
    beta0 = 1.75
    alpha_likelihood = 2.3
    beta = rvs.Gamma(alpha=alpha0, beta=beta0)
    x = rvs.Gamma(alpha=alpha_likelihood, beta=alpha_likelihood*beta,
                  value=x_data)

    beta_cond = conj.complete_conditional(beta, [beta, x])

    self.assertIsInstance(beta_cond, rvs.Gamma)

    sess = tf.InteractiveSession()
    alpha_val, beta_val = sess.run([beta_cond.alpha, beta_cond.beta],
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

    mu_cond = conj.complete_conditional(mu, [mu, x])
    self.assertIsInstance(mu_cond, rvs.Normal)

    sess = tf.InteractiveSession()
    mu_val, sigma_val = sess.run([mu_cond.mu, mu_cond.sigma], {x: x_data})

    self.assertAllClose(sigma_val, (1. / sigma0**2 +
                                    len(x_data) / sigma_likelihood**2) ** -0.5)
    self.assertAllClose(mu_val,
                        sigma_val**2 * (mu0 / sigma0**2 +
                                        1./sigma_likelihood**2 * x_data.sum()))

  def test_normal_normal_scaled(self):
    x_data = np.array([0.1, 0.5, 3.3, 2.7])

    mu0 = 0.3
    sigma0 = 2.1
    sigma_likelihood = 1.2
    c = 2.

    mu = rvs.Normal(mu0, sigma0)
    x = rvs.Normal(c * mu, sigma_likelihood, sample_shape=len(x_data))

    mu_cond = conj.complete_conditional(mu, [mu, x])
    self.assertIsInstance(mu_cond, rvs.Normal)

    sess = tf.InteractiveSession()
    mu_val, sigma_val = sess.run([mu_cond.mu, mu_cond.sigma], {x: x_data})

    self.assertAllClose(sigma_val,
                        (1. / sigma0**2 +
                         c**2 * len(x_data) / sigma_likelihood**2) ** -0.5)
    self.assertAllClose(mu_val,
                        sigma_val**2 * (mu0 / sigma0**2 +
                                        c/sigma_likelihood**2 * x_data.sum()))

  def test_dirichlet_categorical(self):
    x_data = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], np.int32)
    N = x_data.shape[0]
    D = x_data.max() + 1

    alpha = np.zeros([D]).astype(np.float32) + 2.
    sample_shape = (N,)

    theta = rvs.Dirichlet(alpha)
    x = rvs.Categorical(p=theta, sample_shape=sample_shape)

    blanket = [theta, x]
    theta_cond = conj.complete_conditional(theta, blanket)

    sess = tf.InteractiveSession()
    alpha_val = sess.run(theta_cond.alpha, {x: x_data})

    self.assertAllClose(alpha_val, np.array([6., 5., 4., 3.], np.float32))


  def test_bernoulli_mog(self):
    x_val = np.array([1.1, 1.2, 2.1, 4.4, 7.3, 5.5, 6.8], np.float32)
    z_val = np.array([0, 0, 0, 1, 1, 1, 1], np.int32)
    N = x_val.shape[0]

    pi = 0.5
    prior_sigma = 4.
    mu0 = rvs.Normal(0., prior_sigma)
    mu1 = rvs.Normal(0., prior_sigma)
    sigmasq = 2.**2
    z = rvs.Bernoulli(p=pi, sample_shape=N)
    f_z = tf.cast(z.value(), np.float32)
    x_mean = f_z * mu1 + (1. - f_z) * mu0
    x = rvs.Normal(x_mean, tf.sqrt(sigmasq))
    
    blanket = [z, x, mu0, mu1]
    mu0_cond = conj.complete_conditional(mu0, blanket)
    mu1_cond = conj.complete_conditional(mu1, blanket)

    sess = tf.InteractiveSession()
    mu0_mu, mu0_sigma, mu1_mu, mu1_sigma = sess.run([mu0_cond.mu,
                                                     mu0_cond.sigma,
                                                     mu1_cond.mu,
                                                     mu1_cond.sigma],
                                                    {z.value(): z_val,
                                                     x.value(): x_val})
    true_sigmasq_0 = (1./sigmasq * (z_val==0).sum() + 1./prior_sigma**2)**-1
    true_sigmasq_1 = (1./sigmasq * (z_val==1).sum() + 1./prior_sigma**2)**-1
    true_mu_0 = 1./sigmasq * x_val[z_val == 0].sum() * true_sigmasq_0
    true_mu_1 = 1./sigmasq * x_val[z_val == 1].sum() * true_sigmasq_1
    self.assertAllClose(mu0_sigma**2, true_sigmasq_0)
    self.assertAllClose(mu1_sigma**2, true_sigmasq_1)
    self.assertAllClose(mu0_mu, true_mu_0)
    self.assertAllClose(mu1_mu, true_mu_1)


if __name__ == '__main__':
  tf.test.main()
