"""Check that inference classes run without error on a Bayesian neural net."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Empirical, Bernoulli, Normal


def neural_network(x, W_1, W_2, W_3, b_1, b_2):
  h = tf.tanh(tf.matmul(x, W_1) + b_1)
  h = tf.tanh(tf.matmul(h, W_2) + b_2)
  h = tf.matmul(h, W_3)
  return tf.reshape(h, [-1])


class test_inference_bayesian_nn_class(tf.test.TestCase):

  def _test(self):
    X_train = np.zeros([500, 100])
    y_train = np.zeros(500)

    N = X_train.shape[0]  # number of data points
    D = X_train.shape[1]  # number of features

    W_1 = Normal(loc=tf.zeros([D, 20]), scale=tf.ones([D, 20]) * 100)
    W_2 = Normal(loc=tf.zeros([20, 15]), scale=tf.ones([20, 15]) * 100)
    W_3 = Normal(loc=tf.zeros([15, 1]), scale=tf.ones([15, 1]) * 100)
    b_1 = Normal(loc=tf.zeros(20), scale=tf.ones(20) * 100)
    b_2 = Normal(loc=tf.zeros(15), scale=tf.ones(15) * 100)

    X = tf.placeholder(tf.float32, [N, D])
    y = Bernoulli(logits=neural_network(X, W_1, W_2, W_3, b_1, b_2))
    return N, D, W_1, W_2, W_3, b_1, b_2, X, y, X_train, y_train

  def test_gan_inference(self):
    with self.test_session():
      N, D, W_1, W_2, W_3, b_1, b_2, X, y, X_train, y_train = self._test()

      with tf.variable_scope("Gen"):
        theta = tf.get_variable("theta", [1])
        y = tf.cast(y, tf.float32) * theta

      def discriminator(x):
        w = tf.get_variable("w", [1])
        return w * tf.cast(x, tf.float32)

      inference = ed.GANInference(
          data={y: tf.cast(y_train, tf.float32), X: X_train},
          discriminator=discriminator)
      inference.run(n_iter=1)

  def test_wgan_inference(self):
    with self.test_session():
      N, D, W_1, W_2, W_3, b_1, b_2, X, y, X_train, y_train = self._test()

      with tf.variable_scope("Gen"):
        theta = tf.get_variable("theta", [1])
        y = tf.cast(y, tf.float32) * theta

      def discriminator(x):
        w = tf.get_variable("w", [1])
        return w * tf.cast(x, tf.float32)

      inference = ed.WGANInference(
          data={y: tf.cast(y_train, tf.float32), X: X_train},
          discriminator=discriminator)
      inference.run(n_iter=1)

  def test_hmc(self):
    with self.test_session():
      N, D, W_1, W_2, W_3, b_1, b_2, X, y, X_train, y_train = self._test()

      T = 1  # number of MCMC samples
      qW_1 = Empirical(params=tf.Variable(tf.random_normal([T, D, 20])))
      qW_2 = Empirical(params=tf.Variable(tf.random_normal([T, 20, 15])))
      qW_3 = Empirical(params=tf.Variable(tf.random_normal([T, 15, 1])))
      qb_1 = Empirical(params=tf.Variable(tf.random_normal([T, 20])))
      qb_2 = Empirical(params=tf.Variable(tf.random_normal([T, 15])))

      inference = ed.HMC(
          {W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2, W_3: qW_3},
          data={y: y_train, X: X_train})
      inference.run()

  def test_metropolis_hastings(self):
    with self.test_session():
      N, D, W_1, W_2, W_3, b_1, b_2, X, y, X_train, y_train = self._test()

      T = 1  # number of MCMC samples
      qW_1 = Empirical(params=tf.Variable(tf.random_normal([T, D, 20])))
      qW_2 = Empirical(params=tf.Variable(tf.random_normal([T, 20, 15])))
      qW_3 = Empirical(params=tf.Variable(tf.random_normal([T, 15, 1])))
      qb_1 = Empirical(params=tf.Variable(tf.random_normal([T, 20])))
      qb_2 = Empirical(params=tf.Variable(tf.random_normal([T, 15])))

      inference = ed.MetropolisHastings(
          {W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2, W_3: qW_3},
          {W_1: W_1, b_1: b_1, W_2: W_2, b_2: b_2, W_3: W_3},
          data={y: y_train, X: X_train})
      inference.run()

  def test_sgld(self):
    with self.test_session():
      N, D, W_1, W_2, W_3, b_1, b_2, X, y, X_train, y_train = self._test()

      T = 1  # number of MCMC samples
      qW_1 = Empirical(params=tf.Variable(tf.random_normal([T, D, 20])))
      qW_2 = Empirical(params=tf.Variable(tf.random_normal([T, 20, 15])))
      qW_3 = Empirical(params=tf.Variable(tf.random_normal([T, 15, 1])))
      qb_1 = Empirical(params=tf.Variable(tf.random_normal([T, 20])))
      qb_2 = Empirical(params=tf.Variable(tf.random_normal([T, 15])))

      inference = ed.SGLD(
          {W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2, W_3: qW_3},
          data={y: y_train, X: X_train})
      inference.run()

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
