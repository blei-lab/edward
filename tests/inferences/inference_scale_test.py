from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_inference_scale_class(tf.test.TestCase):

  def test_scale_0d(self):
    with self.test_session():
      N = 10
      M = 5
      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones(M) * mu, scale=tf.ones(M))

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      x_ph = tf.placeholder(tf.float32, [M])
      inference = ed.KLqp({mu: qmu}, data={x: x_ph})
      inference.initialize(scale={x: float(N) / M})
      self.assertAllEqual(inference.scale[x], float(N) / M)

  def test_scale_1d(self):
    with self.test_session():
      N = 10
      M = 5
      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones(M) * mu, scale=tf.ones(M))

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      x_ph = tf.placeholder(tf.float32, [M])
      inference = ed.KLqp({mu: qmu}, data={x: x_ph})
      inference.initialize(scale={x: tf.range(M, dtype=tf.float32)})
      self.assertAllEqual(inference.scale[x].eval(), np.arange(M))

if __name__ == '__main__':
  tf.test.main()
