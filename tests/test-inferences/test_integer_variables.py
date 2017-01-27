"""Test that integer variables are handled properly during initialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Categorical


def neural_network(x, W_0, W_1, b_0, b_1):
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.tanh(tf.matmul(h, W_1) + b_1)
  return h


class test_integer_init(tf.test.TestCase):

  def test_integer(self):
    tf.InteractiveSession()     # Eliminates graph dependencies between tests.
    X_train = np.zeros([100, 10]).astype(np.float32)
    y_train = np.zeros(100).astype(np.int32)
    N, D = X_train.shape
    n_hidden = 10
    K = 10
    W_0 = Normal(mu=tf.zeros([D, n_hidden]), sigma=tf.ones([D, n_hidden]))
    W_1 = Normal(mu=tf.zeros([n_hidden, K]), sigma=tf.ones([n_hidden, K]))
    b_0 = Normal(mu=tf.zeros(n_hidden), sigma=tf.ones(n_hidden))
    b_1 = Normal(mu=tf.zeros(K), sigma=tf.ones(K))

    y = Categorical(logits=neural_network(X_train, W_0, W_1, b_0, b_1))

    softplus = tf.nn.softplus
    qW_0 = Normal(mu=tf.Variable(tf.random_normal([D, n_hidden])),
                  sigma=softplus(tf.Variable(tf.random_normal([D, n_hidden]))))
    qW_1 = Normal(mu=tf.Variable(tf.random_normal([n_hidden, K])),
                  sigma=softplus(tf.Variable(tf.random_normal([n_hidden, K]))))
    qb_0 = Normal(mu=tf.Variable(tf.random_normal([n_hidden])),
                  sigma=softplus(tf.Variable(tf.random_normal([n_hidden]))))
    qb_1 = Normal(mu=tf.Variable(tf.random_normal([K])),
                  sigma=softplus(tf.Variable(tf.random_normal([K]))))
    inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1},
                        data={y: y_train})
    inference.run()


if __name__ == '__main__':
  tf.test.main()
