from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward import models as rvs
from edward.inferences import conjugacy as conj
from edward.inferences.conjugacy import simplify as simplify


class test_simplify_class(tf.test.TestCase):

  def test_mul_n(self):
    a = tf.constant(1.)
    b = tf.constant(2.)
    c = tf.constant(3.)
    ab = simplify._mul_n([a, b])
    abc = simplify._mul_n([a, b, c])
    
    sess = tf.InteractiveSession()
    self.assertEquals(sess.run(ab), 2.)
    self.assertEquals(sess.run(abc), 6.)

  def test_as_float(self):
    self.assertEquals(simplify.as_float(1), 1.)
    self.assertIsNone(simplify.as_float('one'))

  def test_identity_op_simplify(self):
    expr = ('#Identity', ('#Mul', ('#Identity', ('#x',)),
                          ('#Identity', ('3.7',))))
    did_something, new_expr = simplify.identity_op_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Mul', ('#x',), ('3.7',)))
    did_something, new_expr = simplify.power_op_simplify(new_expr)
    self.assertFalse(did_something)
    

  def test_pow_simplify_and_power_op_simplify(self):
    expr = ('#Square', ('#Reciprocal', ('#Sqrt', ('#x',))))
    did_something, new_expr = simplify.power_op_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr,
                      ('#Pow2.0000e+00',
                       ('#Pow-1.0000e+00', ('#Pow5.0000e-01', ('#x',)))))
    did_something, new_expr = simplify.power_op_simplify(new_expr)
    self.assertFalse(did_something)

    did_something, new_expr = simplify.pow_simplify(new_expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Pow-1.0000e+00', ('#x',)))
    did_something, new_expr = simplify.pow_simplify(new_expr)
    self.assertFalse(did_something)

  def test_log_pow_simplify(self):
    expr = ('#Log', ('#Pow2.3000e+01', ('#x',)))
    did_something, new_expr = simplify.log_pow_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Mul', ('2.3000e+01',), ('#Log', ('#x',))))
    did_something, new_expr = simplify.log_pow_simplify(new_expr)
    self.assertFalse(did_something)

  def test_log_mul_simplify(self):
    expr = ('#Log', ('#Mul', ('3',), ('4.2',), ('1.2e+01',), ('#x',)))
    did_something, new_expr = simplify.log_mul_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Add', ('#Log', ('3',)),
                                 ('#Log', ('4.2',)), ('#Log', ('1.2e+01',)),
                                 ('#Log', ('#x',))))
    did_something, new_expr = simplify.log_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_pow_mul_simplify(self):
    expr = ('#Pow2.1', ('#Mul', ('3',), ('4.',), ('1.2e+01',)))
    did_something, new_expr = simplify.pow_mul_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Mul', ('#Pow2.1', ('3',)),
                                 ('#Pow2.1', ('4.',)),
                                 ('#Pow2.1', ('1.2e+01',))))
    did_something, new_expr = simplify.pow_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_add_simplify(self):
    expr = ('#Mul', ('#Add', ('3.',), ('2.',)),
            ('#Add', ('4.',), ('5.',)))
    did_something, new_expr = simplify.mul_add_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Add', ('#Add', ('#Mul', ('3.',), ('4.',)),
                                          ('#Mul', ('3.',), ('5.',))),
                                 ('#Add', ('#Mul', ('2.',), ('4.',)),
                                  ('#Mul', ('2.',), ('5.',)))))
    did_something, new_expr = simplify.pow_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_add_add_simplify(self):
    expr = ('#Add', ('3.',), ('#Add', ('4.',), ('5.',), ('#Add', ('6.',))))
    did_something, new_expr = simplify.add_add_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Add', ('3.',), ('4.',), ('5.',), ('6.',)))
    did_something, new_expr = simplify.add_add_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_mul_simplify(self):
    expr = ('#Mul', ('3.',), ('#Mul', ('4.',), ('5.',), ('#Mul', ('6.',))))
    did_something, new_expr = simplify.mul_mul_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Mul', ('3.',), ('4.',), ('5.',), ('6.',)))
    did_something, new_expr = simplify.mul_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_one_simplify(self):
    expr = ('#Mul', ('3.',), ('1.',), ('4.',), ('5.',), ('6.',), ('1.',))
    did_something, new_expr = simplify.mul_one_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Mul', ('3.',), ('4.',), ('5.',), ('6.',)))
    did_something, new_expr = simplify.mul_one_simplify(new_expr)
    self.assertFalse(did_something)

  def test_add_zero_simplify(self):
    expr = ('#Add', ('3.',), ('0.',), ('4.',), ('5.',), ('6.',), ('0.',))
    did_something, new_expr = simplify.add_zero_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Add', ('3.',), ('4.',), ('5.',), ('6.',)))
    did_something, new_expr = simplify.add_zero_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_zero_simplify(self):
    expr = ('#Mul', ('3.',), ('0.',), ('5.',), ('6.',), ('1.',))
    did_something, new_expr = simplify.mul_zero_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('0',))
    did_something, new_expr = simplify.mul_zero_simplify(new_expr)
    self.assertFalse(did_something)

  def test_expr_contains(self):
    expr = ('#Add', ('#Mul', ('1.5', ('#Pow1.2000e+00', ('#x',)))),
            ('#Mul', ('1.2',), ('7',)))
    self.assertTrue(simplify.expr_contains(expr, '#x'))
    self.assertFalse(simplify.expr_contains(expr, '#Pow'))
    self.assertTrue(simplify.expr_contains(expr, '#Pow1.2000e+00'))
    self.assertTrue(simplify.expr_contains(expr, '#Mul'))
    self.assertTrue(simplify.expr_contains(expr, '#Add'))
    self.assertTrue(simplify.expr_contains(expr[1], '#x'))
    self.assertFalse(simplify.expr_contains(expr[2], '#x'))

  def test_add_const_simplify(self):
    expr = ('#Add', ('#Mul', ('1.5', ('#Pow1.2000e+00', ('#x',)))),
            ('#Mul', ('1.2',), ('7',)))
    did_something, new_expr = simplify.add_const_simplify(expr)
    self.assertTrue(did_something)
    self.assertEquals(new_expr, ('#Add', ('#Mul',
                                          ('1.5',
                                           ('#Pow1.2000e+00', ('#x',))))))
    did_something, new_expr = simplify.add_const_simplify(new_expr)
    self.assertFalse(did_something)


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


if __name__ == '__main__':
  tf.test.main()
