from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import MultivariateNormalDiag, MultivariateNormalTriL, Normal


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
    w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
    b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
    y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

    return N, D, w_true, X_train, y_train, X, w, b, y

  def _test(self, sess, qw, qb, w_true):
    qw_loc, qb_loc, qw_det_covariance, qb_det_covariance = \
        sess.run([qw.loc, qb.loc,
                  tf.square(qw.scale.determinant()),
                  tf.square(qb.scale.determinant())])

    self.assertAllClose(qw_loc, w_true, atol=0.5)
    self.assertAllClose(qb_loc, np.array([0.0]), atol=0.5)
    self.assertAllClose(qw_det_covariance, 0.0, atol=0.1)
    self.assertAllClose(qb_det_covariance, 0.0, atol=0.1)

  def test_list(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE
      inference = ed.Laplace([w, b], data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      qw = inference.latent_vars[w]
      qb = inference.latent_vars[b]
      self._test(sess, qw, qb, w_true)

  def test_multivariate_normal_tril(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE. Initialize scales at identity to verify if we
      # learned an approximately zero determinant.
      qw = MultivariateNormalTriL(
          loc=tf.Variable(tf.random_normal([D])),
          scale_tril=tf.Variable(tf.diag(tf.ones(D))))
      qb = MultivariateNormalTriL(
          loc=tf.Variable(tf.random_normal([1])),
          scale_tril=tf.Variable(tf.diag(tf.ones(1))))

      inference = ed.Laplace({w: qw, b: qb}, data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      self._test(sess, qw, qb, w_true)

  def test_multivariate_normal_diag(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE. Initialize scales at identity to verify if we
      # learned an approximately zero determinant.
      qw = MultivariateNormalDiag(
          loc=tf.Variable(tf.random_normal([D])),
          scale_diag=tf.Variable(tf.ones(D)))
      qb = MultivariateNormalDiag(
          loc=tf.Variable(tf.random_normal([1])),
          scale_diag=tf.Variable(tf.ones(1)))

      inference = ed.Laplace({w: qw, b: qb}, data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      self._test(sess, qw, qb, w_true)
      self.assertAllClose(qw.covariance().eval(),
                          tf.diag(tf.diag_part(qw.covariance())).eval())
      self.assertAllClose(qb.covariance().eval(),
                          tf.diag(tf.diag_part(qb.covariance())).eval())

  def test_normal(self):
    with self.test_session() as sess:
      N, D, w_true, X_train, y_train, X, w, b, y = self._setup()

      # INFERENCE. Initialize scales at identity to verify if we
      # learned an approximately zero determinant.
      qw = Normal(
          loc=tf.Variable(tf.random_normal([D])),
          scale=tf.Variable(tf.ones(D)))
      qb = Normal(
          loc=tf.Variable(tf.random_normal([1])),
          scale=tf.Variable(tf.ones(1)))

      inference = ed.Laplace({w: qw, b: qb}, data={X: X_train, y: y_train})
      inference.run(n_iter=100)

      qw_loc, qb_loc, qw_scale_det, qb_scale_det = \
          sess.run([qw.loc, qb.loc,
                    tf.reduce_prod(qw.scale), tf.reduce_prod(qb.scale)])
      self.assertAllClose(qw_loc, w_true, atol=0.5)
      self.assertAllClose(qb_loc, np.array([0.0]), atol=0.5)
      self.assertAllClose(qw_scale_det, 0.0, atol=0.1)
      self.assertAllClose(qb_scale_det, 0.0, atol=0.1)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
