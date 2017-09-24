from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Gamma, Normal, PointMass, TransformedDistribution


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

      # Check variational approximation on constrained space has same
      # mean and variance as target distribution.
      n_samples = 10000
      x_mean, x_var = tf.nn.moments(x.sample(n_samples), 0)
      qx_transformed = TransformedDistribution(
          distribution=qx,
          bijector=tf.contrib.distributions.bijectors.Softplus())
      qx_mean, qx_var = tf.nn.moments(qx_transformed.sample(n_samples), 0)
      stats = sess.run([x_mean, qx_mean, x_var, qx_var])
      self.assertAllClose(info_dict['loss'], 0.0, rtol=0.2, atol=0.2)
      self.assertAllClose(stats[0], stats[1], rtol=1e-2, atol=1e-2)
      self.assertAllClose(stats[2], stats[3], rtol=1e-2, atol=1e-2)

  def test_map(self):
    with self.test_session() as sess:
      x = Gamma(2.0, 0.5)
      qx = PointMass(tf.Variable(0.5))

      inference = ed.MAP({x: qx})
      inference.initialize(auto_transform=True, n_iter=1000)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      # Check point estimate on constrained space has same
      # mode as target distribution.
      qx_transformed = TransformedDistribution(
          distribution=qx,
          bijector=tf.contrib.distributions.bijectors.Softplus())
      stats = sess.run([x.mode(), qx_transformed])
      self.assertAllClose(stats[0], stats[1], rtol=1e-5, atol=1e-5)

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

if __name__ == '__main__':
  ed.set_seed(124125)
  tf.test.main()
