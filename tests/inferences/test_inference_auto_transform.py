from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import (Empirical, Gamma, Normal, PointMass,
                           TransformedDistribution)
from edward.util import transform
from tensorflow.contrib.distributions import bijectors


class test_inference_auto_transform_class(tf.test.TestCase):

  def test_auto_transform_true(self):
    with self.test_session() as sess:
      # Match normal || softplus-inverse-normal distribution with
      # automated transformation on latter (assuming it is softplus).
      x = TransformedDistribution(
          distribution=Normal(0.0, 0.5),
          bijector=tf.contrib.distributions.bijectors.Softplus())
      x.support = 'nonnegative'
      qx = Normal(loc=tf.Variable(tf.random_normal([])),
                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

      inference = ed.KLqp({x: qx})
      inference.initialize(auto_transform=True, n_samples=5, n_iter=1000)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      # Check approximation on constrained space has same moments as
      # target distribution.
      n_samples = 10000
      x_mean, x_var = tf.nn.moments(x.sample(n_samples), 0)
      x_unconstrained = inference.transformations[x]
      qx_constrained = transform(qx, bijectors.Invert(x_unconstrained.bijector))
      qx_mean, qx_var = tf.nn.moments(qx_constrained.sample(n_samples), 0)
      stats = sess.run([x_mean, qx_mean, x_var, qx_var])
      self.assertAllClose(info_dict['loss'], 0.0, rtol=0.2, atol=0.2)
      self.assertAllClose(stats[0], stats[1], rtol=1e-1, atol=1e-1)
      self.assertAllClose(stats[2], stats[3], rtol=1e-1, atol=1e-1)

  def test_auto_transform_false(self):
    with self.test_session():
      # Match normal || softplus-inverse-normal distribution without
      # automated transformation; it should fail.
      x = TransformedDistribution(
          distribution=Normal(0.0, 0.5),
          bijector=tf.contrib.distributions.bijectors.Softplus())
      x.support = 'nonnegative'
      qx = Normal(loc=tf.Variable(tf.random_normal([])),
                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

      inference = ed.KLqp({x: qx})
      inference.initialize(auto_transform=False, n_samples=5, n_iter=150)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      self.assertAllEqual(info_dict['loss'], np.nan)

  def test_map_custom(self):
    with self.test_session() as sess:
      x = Gamma(2.0, 0.5)
      qx = PointMass(tf.nn.softplus(tf.Variable(0.5)))

      inference = ed.MAP({x: qx})
      inference.initialize(auto_transform=True, n_iter=500)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      # Check approximation on constrained space has same mode as
      # target distribution.
      stats = sess.run([x.mode(), qx])
      self.assertAllClose(stats[0], stats[1], rtol=1e-5, atol=1e-5)

  def test_map_default(self):
    with self.test_session() as sess:
      x = Gamma(2.0, 0.5)

      inference = ed.MAP([x])
      inference.initialize(auto_transform=True, n_iter=500)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      # Check approximation on constrained space has same mode as
      # target distribution.
      qx = inference.latent_vars[x]
      stats = sess.run([x.mode(), qx])
      self.assertAllClose(stats[0], stats[1], rtol=1e-5, atol=1e-5)

  def test_laplace_default(self):
    with self.test_session() as sess:
      x = Gamma([2.0], [0.5])

      inference = ed.Laplace([x])
      optimizer = tf.train.AdamOptimizer(0.2)
      inference.initialize(auto_transform=True, n_iter=500)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      # Check approximation on constrained space has same mode as
      # target distribution.
      qx = inference.latent_vars[x]
      stats = sess.run([x.mode(), qx.mean()])
      self.assertAllClose(stats[0], stats[1], rtol=1e-5, atol=1e-5)

  def test_hmc_custom(self):
    with self.test_session() as sess:
      x = TransformedDistribution(
          distribution=Normal(1.0, 1.0),
          bijector=tf.contrib.distributions.bijectors.Softplus())
      x.support = 'nonnegative'
      qx = Empirical(tf.Variable(tf.random_normal([1000])))

      inference = ed.HMC({x: qx})
      inference.initialize(auto_transform=True, step_size=0.8)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      # Check approximation on constrained space has same moments as
      # target distribution.
      n_samples = 10000
      x_unconstrained = inference.transformations[x]
      qx_constrained = Empirical(x_unconstrained.bijector.inverse(qx.params))
      x_mean, x_var = tf.nn.moments(x.sample(n_samples), 0)
      qx_mean, qx_var = tf.nn.moments(qx_constrained.params[500:], 0)
      stats = sess.run([x_mean, qx_mean, x_var, qx_var])
      self.assertAllClose(stats[0], stats[1], rtol=1e-1, atol=1e-1)
      self.assertAllClose(stats[2], stats[3], rtol=1e-1, atol=1e-1)

  def test_hmc_default(self):
    with self.test_session() as sess:
      x = TransformedDistribution(
          distribution=Normal(1.0, 1.0),
          bijector=tf.contrib.distributions.bijectors.Softplus())
      x.support = 'nonnegative'

      inference = ed.HMC([x])
      inference.initialize(auto_transform=True, step_size=0.8)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)

      # Check approximation on constrained space has same moments as
      # target distribution.
      n_samples = 10000
      x_unconstrained = inference.transformations[x]
      qx = inference.latent_vars[x_unconstrained]
      qx_constrained = Empirical(x_unconstrained.bijector.inverse(qx.params))
      x_mean, x_var = tf.nn.moments(x.sample(n_samples), 0)
      qx_mean, qx_var = tf.nn.moments(qx_constrained.params[500:], 0)
      stats = sess.run([x_mean, qx_mean, x_var, qx_var])
      self.assertAllClose(stats[0], stats[1], rtol=1e-1, atol=1e-1)
      self.assertAllClose(stats[2], stats[3], rtol=1e-1, atol=1e-1)

if __name__ == '__main__':
  ed.set_seed(124125)
  tf.test.main()
