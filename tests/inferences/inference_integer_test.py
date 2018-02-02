"""Test that integer variables are handled properly during initialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Categorical


def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.nn.relu(tf.matmul(x, W_0) + b_0)
  h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
  return h


class test_integer_init(tf.test.TestCase):

  def test(self):
    with self.test_session():
      X_train = np.zeros([100, 10], dtype=np.float32)
      y_train = np.zeros(100, dtype=np.int32)

      N, D = X_train.shape
      H = 10  # number of hidden units
      K = 10  # number of classes

      W_0 = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
      W_1 = Normal(loc=tf.zeros([H, K]), scale=tf.ones([H, K]))
      b_0 = Normal(loc=tf.zeros(H), scale=tf.ones(H))
      b_1 = Normal(loc=tf.zeros(K), scale=tf.ones(K))

      y = Categorical(logits=neural_network(X_train, W_0, W_1, b_0, b_1))

      qW_0 = Normal(
          loc=tf.Variable(tf.random_normal([D, H])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, H]))))
      qW_1 = Normal(
          loc=tf.Variable(tf.random_normal([H, K])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([H, K]))))
      qb_0 = Normal(
          loc=tf.Variable(tf.random_normal([H])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([H]))))
      qb_1 = Normal(
          loc=tf.Variable(tf.random_normal([K])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

      inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1},
                          data={y: y_train})
      inference.run(n_iter=1)

if __name__ == '__main__':
  tf.test.main()
