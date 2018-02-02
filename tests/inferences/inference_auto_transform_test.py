from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import (Empirical, Gamma, Normal, PointMass,
                           TransformedDistribution, Beta, Bernoulli)
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
      qx_constrained_params = x_unconstrained.bijector.inverse(qx.params)
      x_mean, x_var = tf.nn.moments(x.sample(n_samples), 0)
      qx_mean, qx_var = tf.nn.moments(qx_constrained_params[500:], 0)
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
      n_samples = 1000
      qx_constrained = inference.latent_vars[x]
      x_mean, x_var = tf.nn.moments(x.sample(n_samples), 0)
      qx_mean, qx_var = tf.nn.moments(qx_constrained.params[500:], 0)
      stats = sess.run([x_mean, qx_mean, x_var, qx_var])
      self.assertAllClose(stats[0], stats[1], rtol=1e-1, atol=1e-1)
      self.assertAllClose(stats[2], stats[3], rtol=1e-1, atol=1e-1)

  def test_hmc_betabernoulli(self):
    """Do we correctly handle dependencies of transformed variables?"""

    with self.test_session() as sess:
      # model
      z = Beta(1., 1., name="z")
      xs = Bernoulli(probs=z, sample_shape=10)
      x_obs = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 0, 1], dtype=np.int32)

      # inference
      qz_samples = tf.Variable(tf.random_uniform(shape=(1000,)))
      qz = ed.models.Empirical(params=qz_samples, name="z_posterior")
      inference_hmc = ed.inferences.HMC({z: qz}, data={xs: x_obs})
      inference_hmc.run(step_size=1.0, n_steps=5, auto_transform=True)

      # check that inferred posterior mean/variance is close to
      # that of the exact Beta posterior
      z_unconstrained = inference_hmc.transformations[z]
      qz_constrained = z_unconstrained.bijector.inverse(qz_samples)
      qz_mean, qz_var = sess.run(tf.nn.moments(qz_constrained, 0))

      true_posterior = Beta(1. + np.sum(x_obs), 1. + np.sum(1 - x_obs))
      pz_mean, pz_var = sess.run((true_posterior.mean(),
                                  true_posterior.variance()))
      self.assertAllClose(qz_mean, pz_mean, rtol=5e-2, atol=5e-2)
      self.assertAllClose(qz_var, pz_var, rtol=1e-2, atol=1e-2)

  def test_klqp_betabernoulli(self):
    with self.test_session() as sess:
      # model
      z = Beta(1., 1., name="z")
      xs = Bernoulli(probs=z, sample_shape=10)
      x_obs = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 0, 1], dtype=np.int32)

      # inference
      qz_mean = tf.get_variable("qz_mean",
                                initializer=tf.random_normal(()))
      qz_std = tf.nn.softplus(tf.get_variable(name="qz_prestd",
                                              initializer=tf.random_normal(())))
      qz_unconstrained = ed.models.Normal(
          loc=qz_mean, scale=qz_std, name="z_posterior")

      inference_klqp = ed.inferences.KLqp(
          {z: qz_unconstrained}, data={xs: x_obs})
      inference_klqp.run(n_iter=500, auto_transform=True)

      z_unconstrained = inference_klqp.transformations[z]
      qz_constrained = z_unconstrained.bijector.inverse(
          qz_unconstrained.sample(1000))
      qz_mean, qz_var = sess.run(tf.nn.moments(qz_constrained, 0))

      true_posterior = Beta(np.sum(x_obs) + 1., np.sum(1 - x_obs) + 1.)
      pz_mean, pz_var = sess.run((true_posterior.mean(),
                                  true_posterior.variance()))
      self.assertAllClose(qz_mean, pz_mean, rtol=5e-2, atol=5e-2)
      self.assertAllClose(qz_var, pz_var, rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
  ed.set_seed(124125)
  tf.test.main()
