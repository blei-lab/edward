from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import \
    MultivariateNormalCholesky, MultivariateNormalDiag, \
    MultivariateNormalFull, Normal


def build_toy_dataset(N, w, noise_std=0.1):
  D = len(w)
  x = np.random.randn(N, D).astype(np.float32)
  y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
  return x, y


class test_laplace_class(tf.test.TestCase):

  def _setup(self):
    N = 250  # number of data points
    D = 5  # number of features

    # DATA
    w_true = np.ones(D) * 5.0
    X_train, y_train = build_toy_dataset(N, w_true)

    # MODEL
    X = tf.placeholder(tf.float32, [N, D])
    w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
    b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
    y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

    return N, D, w_true, X_train, y_train, X, w, b, y

  def _test(self, sess, qw, qb, w_true):
    qw_mu, qb_mu, qw_sigma_det, qb_sigma_det = \
        sess.run([qw.mu, qb.mu, qw.sigma_det(), qb.sigma_det()])

    self.assertAllClose(qw_mu, w_true, atol=0.5)
    self.assertAllClose(qb_mu, np.array([0.0]), atol=0.5)
    self.assertAllClose(qw_sigma_det, 0.0, atol=0.1)
    self.assertAllClose(qb_sigma_det, 0.0, atol=0.1)

  def test_list(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE
      inference = ed.Laplace([w, b], data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      qw = inference.latent_vars[w]
      qb = inference.latent_vars[b]
      self._test(sess, qw, qb, w_true)

  def test_multivariate_normal_cholesky(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE. Initialize sigma's at identity to verify if we
      # learned an approximately zero determinant.
      qw = MultivariateNormalCholesky(
          mu=tf.Variable(tf.random_normal([D])),
          chol=tf.Variable(tf.diag(tf.ones(D))))
      qb = MultivariateNormalCholesky(
          mu=tf.Variable(tf.random_normal([1])),
          chol=tf.Variable(tf.diag(tf.ones(1))))

      inference = ed.Laplace({w: qw, b: qb}, data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      self._test(sess, qw, qb, w_true)

  def test_multivariate_normal_diag(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE. Initialize sigma's at identity to verify if we
      # learned an approximately zero determinant.
      qw = MultivariateNormalDiag(
          mu=tf.Variable(tf.random_normal([D])),
          diag_stdev=tf.Variable(tf.ones(D)))
      qb = MultivariateNormalDiag(
          mu=tf.Variable(tf.random_normal([1])),
          diag_stdev=tf.Variable(tf.ones(1)))

      inference = ed.Laplace({w: qw, b: qb}, data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      self._test(sess, qw, qb, w_true)
      self.assertAllClose(qw.sigma.eval(),
                          tf.diag(tf.diag_part(qw.sigma)).eval())
      self.assertAllClose(qb.sigma.eval(),
                          tf.diag(tf.diag_part(qb.sigma)).eval())

  def test_multivariate_normal_full(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE. Initialize sigma's at identity to verify if we
      # learned an approximately zero determinant.
      qw = MultivariateNormalFull(
          mu=tf.Variable(tf.random_normal([D])),
          sigma=tf.Variable(tf.diag(tf.ones(D))))
      qb = MultivariateNormalFull(
          mu=tf.Variable(tf.random_normal([1])),
          sigma=tf.Variable(tf.diag(tf.ones(1))))

      inference = ed.Laplace({w: qw, b: qb}, data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      self._test(sess, qw, qb, w_true)

  def test_normal(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE. Initialize sigma's at identity to verify if we
      # learned an approximately zero determinant.
      qw = Normal(
          mu=tf.Variable(tf.random_normal([D])),
          sigma=tf.Variable(tf.ones(D)))
      qb = Normal(
          mu=tf.Variable(tf.random_normal([1])),
          sigma=tf.Variable(tf.ones(1)))

      inference = ed.Laplace({w: qw, b: qb}, data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      qw_mu, qb_mu, qw_sigma_det, qb_sigma_det = \
          sess.run([qw.mu, qb.mu,
                    tf.reduce_prod(qw.sigma), tf.reduce_prod(qb.sigma)])
      self.assertAllClose(qw_mu, w_true, atol=0.5)
      self.assertAllClose(qb_mu, np.array([0.0]), atol=0.5)
      self.assertAllClose(qw_sigma_det, 0.0, atol=0.1)
      self.assertAllClose(qb_sigma_det, 0.0, atol=0.1)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
