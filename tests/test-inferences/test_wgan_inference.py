from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from tensorflow.contrib import slim


def next_batch(M):
  samples = np.random.normal(4.0, 0.1, M)
  samples.sort()
  return samples


def discriminative_network(x):
  """Outputs probability in logits."""
  h0 = slim.fully_connected(x, 10, activation_fn=tf.nn.relu)
  return slim.fully_connected(h0, 1, activation_fn=None)


class test_wgan_class(tf.test.TestCase):

  def test_normal_clip(self):
    with self.test_session() as sess:
      # DATA
      M = 12  # batch size during training
      x_ph = tf.placeholder(tf.float32, [M, 1])

      # MODEL
      with tf.variable_scope("Gen"):
        theta = tf.Variable(0.0)
        x = Normal(theta, 0.1, sample_shape=[M, 1])

      # INFERENCE
      inference = ed.WGANInference(
          data={x: x_ph}, discriminator=discriminative_network)
      inference.initialize(penalty=None, clip=0.01, n_iter=500)
      tf.global_variables_initializer().run()

      for _ in range(inference.n_iter):
        x_data = next_batch(M).reshape([M, 1])
        for _ in range(5):
          info_dict_d = inference.update(feed_dict={x_ph: x_data},
                                         variables="Disc")

        inference.update(feed_dict={x_ph: x_data}, variables="Gen")

      self.assertAllClose(theta.eval(), 4.0, rtol=1.0, atol=1.0)

  def test_normal_penalty(self):
    with self.test_session() as sess:
      # DATA
      M = 12  # batch size during training
      x_ph = tf.placeholder(tf.float32, [M, 1])

      # MODEL
      with tf.variable_scope("Gen"):
        theta = tf.Variable(0.0)
        x = Normal(theta, 0.1, sample_shape=[M, 1])

      # INFERENCE
      inference = ed.WGANInference(
          data={x: x_ph}, discriminator=discriminative_network)
      inference.initialize(penalty=0.1, n_iter=500)
      tf.global_variables_initializer().run()

      for _ in range(inference.n_iter):
        x_data = next_batch(M).reshape([M, 1])
        for _ in range(5):
          info_dict_d = inference.update(feed_dict={x_ph: x_data},
                                         variables="Disc")

        inference.update(feed_dict={x_ph: x_data}, variables="Gen")

      # CRITICISM
      self.assertAllClose(theta.eval(), 4.0, rtol=1.0, atol=1.0)

if __name__ == '__main__':
  ed.set_seed(12451)
  tf.test.main()
