from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward import models as rvs
from edward.inferences import conjugacy as conj  


class test_conjugacy_class(tf.test.TestCase):
  
#   def test_beta_bernoulli(self):
#     x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

#     a0 = 0.5
#     b0 = 1.5
#     pi = rvs.Beta(a=a0, b=b0)
#     x = rvs.Bernoulli(p=pi, sample_shape=10)

#     pi_cond = conj.complete_conditional(pi, [pi, x])

#     self.assertIsInstance(pi_cond, rvs.Beta)

#     sess = tf.InteractiveSession()
#     a_val, b_val = sess.run([pi_cond.a, pi_cond.b], {x: x_data})

#     self.assertAllClose(a_val, a0 + x_data.sum())
#     self.assertAllClose(b_val, b0 + (1-x_data).sum())

#   def test_gamma_poisson(self):
#     x_data = np.array([0, 1, 0, 7, 0, 0, 2, 0, 0, 1])

#     alpha0 = 0.5
#     beta0 = 1.75
#     lam = rvs.Gamma(alpha=alpha0, beta=beta0)
#     x = rvs.Poisson(lam=lam, value=x_data)

#     lam_cond = conj.complete_conditional(lam, [lam, x])

#     self.assertIsInstance(lam_cond, rvs.Gamma)

#     sess = tf.InteractiveSession()
#     alpha_val, beta_val = sess.run([lam_cond.alpha, lam_cond.beta], {x: x_data})
#     self.assertAllClose(alpha_val, alpha0 + x_data.sum())
#     self.assertAllClose(beta_val, beta0 + len(x_data))

#   def test_gamma_gamma(self):
#     x_data = np.array([0.1, 0.5, 3.3, 2.7])

#     alpha0 = 0.5
#     beta0 = 1.75
#     alpha_likelihood = 2.3
#     beta = rvs.Gamma(alpha=alpha0, beta=beta0)
#     x = rvs.Gamma(alpha=alpha_likelihood, beta=beta,
#                   value=x_data)

#     beta_cond = conj.complete_conditional(beta, [beta, x])

#     self.assertIsInstance(beta_cond, rvs.Gamma)

#     sess = tf.InteractiveSession()
#     alpha_val, beta_val = sess.run([beta_cond.alpha, beta_cond.beta],
#                                    {x: x_data})
#     self.assertAllClose(alpha_val, alpha0 + alpha_likelihood * len(x_data))
#     self.assertAllClose(beta_val, beta0 + x_data.sum())

#   def test_unpack(self):
#     a = 3.
#     b = tf.constant([1., 7.3])
#     c = tf.pow(a, 2.2)
#     d = tf.constant(7.9)

#     sess = tf.InteractiveSession()

#     multiplicands = conj._unpack_mul((a * b).op)
#     multiplicand_values = sess.run(multiplicands)
#     self.assertAllClose(multiplicand_values[0], a)
#     self.assertAllClose(multiplicand_values[1], sess.run(b))

#     multiplicands = conj._unpack_mul((a * b * c * d).op)
#     multiplicand_values = sess.run(multiplicands)
#     self.assertAllClose(multiplicand_values[0], a)
#     self.assertAllClose(multiplicand_values[1], sess.run(b))
#     self.assertAllClose(multiplicand_values[2], sess.run(c))
#     self.assertAllClose(multiplicand_values[3], sess.run(d))

#     multiplicands = conj._unpack_mul(((a * b) * (c * d)).op)
#     multiplicand_values = sess.run(multiplicands)
#     self.assertAllClose(multiplicand_values[0], a)
#     self.assertAllClose(multiplicand_values[1], sess.run(b))
#     self.assertAllClose(multiplicand_values[2], sess.run(c))
#     self.assertAllClose(multiplicand_values[3], sess.run(d))

#     terms = conj._unpack_mul_add(((a * b) + (c * d)).op)
#     term_values = sess.run(terms)
#     self.assertAllClose(term_values[0], a)
#     self.assertAllClose(term_values[1], sess.run(b))
#     self.assertAllClose(term_values[2], sess.run(c))
#     self.assertAllClose(term_values[3], sess.run(d))

#   def test_mul_rate_gamma(self):
#     x_data = np.array([0.1, 0.5, 3.3, 2.7])

#     alpha0 = 0.5
#     beta0 = 1.75
#     alpha_likelihood = 2.3
#     beta = rvs.Gamma(alpha=alpha0, beta=beta0)
#     x = rvs.Gamma(alpha=alpha_likelihood, beta=alpha_likelihood*beta,
#                   value=x_data)

#     beta_cond = conj.complete_conditional(beta, [beta, x])

#     self.assertIsInstance(beta_cond, rvs.Gamma)

#     sess = tf.InteractiveSession()
#     alpha_val, beta_val = sess.run([beta_cond.alpha, beta_cond.beta],
#                                    {x: x_data})
#     self.assertAllClose(alpha_val, alpha0 + alpha_likelihood * len(x_data))
#     self.assertAllClose(beta_val, beta0 + alpha_likelihood * x_data.sum())

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


if __name__ == '__main__':
  tf.test.main()
