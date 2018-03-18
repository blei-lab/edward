from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Empirical, Normal


class test_hmc_class(tf.test.TestCase):

  def _test_normal_normal(self, default, dtype):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=tf.constant(0.0, dtype=dtype),
                  scale=tf.constant(1.0, dtype=dtype))
      x = Normal(loc=mu, scale=tf.constant(1.0, dtype=dtype),
                 sample_shape=50)

      n_samples = 2000
      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      if not default:
        qmu = Empirical(params=tf.Variable(tf.ones(n_samples, dtype=dtype)))
        inference = ed.HMC({mu: qmu}, data={x: x_data})
      else:
        inference = ed.HMC([mu], data={x: x_data})
        qmu = inference.latent_vars[mu]
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-1, atol=1e-1)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-1, atol=1e-1)

      old_t, old_n_accept = sess.run([inference.t, inference.n_accept])
      if not default:
        self.assertEqual(old_t, n_samples)
      else:
        self.assertEqual(old_t, 1e4)
      self.assertGreater(old_n_accept, 0.1)
      sess.run(inference.reset)
      new_t, new_n_accept = sess.run([inference.t, inference.n_accept])
      self.assertEqual(new_t, 0)
      self.assertEqual(new_n_accept, 0)

  def _test_linear_regression(self, default, dtype):
    def build_toy_dataset(N, w, noise_std=0.1):
      D = len(w)
      x = np.random.randn(N, D)
      y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
      return x, y

    with self.test_session() as sess:
      N = 40  # number of data points
      D = 10  # number of features

      w_true = np.random.randn(D)
      X_train, y_train = build_toy_dataset(N, w_true)
      X_test, y_test = build_toy_dataset(N, w_true)

      X = tf.placeholder(dtype, [N, D])
      w = Normal(loc=tf.zeros(D, dtype=dtype), scale=tf.ones(D, dtype=dtype))
      b = Normal(loc=tf.zeros(1, dtype=dtype), scale=tf.ones(1, dtype=dtype))
      y = Normal(loc=ed.dot(X, w) + b, scale=0.1 * tf.ones(N, dtype=dtype))

      n_samples = 2000
      if not default:
        qw = Empirical(tf.Variable(tf.zeros([n_samples, D], dtype=dtype)))
        qb = Empirical(tf.Variable(tf.zeros([n_samples, 1], dtype=dtype)))
        inference = ed.HMC({w: qw, b: qb}, data={X: X_train, y: y_train})
      else:
        inference = ed.HMC([w, b], data={X: X_train, y: y_train})
        qw = inference.latent_vars[w]
        qb = inference.latent_vars[b]
      inference.run(step_size=0.01)

      self.assertAllClose(qw.mean().eval(), w_true, rtol=5e-1, atol=5e-1)
      self.assertAllClose(qb.mean().eval(), [0.0], rtol=5e-1, atol=5e-1)

      old_t, old_n_accept = sess.run([inference.t, inference.n_accept])
      if not default:
        self.assertEqual(old_t, n_samples)
      else:
        self.assertEqual(old_t, 1e4)
      self.assertGreater(old_n_accept, 0.1)
      sess.run(inference.reset)
      new_t, new_n_accept = sess.run([inference.t, inference.n_accept])
      self.assertEqual(new_t, 0)
      self.assertEqual(new_n_accept, 0)

  def test_normal_normal(self):
    self._test_normal_normal(True, tf.float32)
    self._test_normal_normal(False, tf.float32)
    self._test_normal_normal(True, tf.float64)
    self._test_normal_normal(False, tf.float64)

  def test_linear_regression(self):
    self._test_linear_regression(True, tf.float32)
    self._test_linear_regression(False, tf.float32)
    self._test_linear_regression(True, tf.float64)
    self._test_linear_regression(False, tf.float64)

  def test_indexedslices(self):
    """Test that gradients accumulate when tf.gradients doesn't return
    tf.Tensor (IndexedSlices)."""
    with self.test_session() as sess:
      N = 10  # number of data points
      K = 2  # number of clusters
      T = 1  # number of MCMC samples

      x_data = np.zeros(N, dtype=np.float32)

      mu = Normal(0.0, 1.0, sample_shape=K)
      c = Categorical(logits=tf.zeros(N))
      x = Normal(tf.gather(mu, c), tf.ones(N))

      qmu = Empirical(params=tf.Variable(tf.ones([T, K])))
      qc = Empirical(params=tf.Variable(tf.ones([T, N])))

      inference = ed.HMC({mu: qmu}, data={x: x_data})
      inference.initialize()

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
