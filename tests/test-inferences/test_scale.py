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
      mu = Normal(mu=0.0, sigma=1.0)
      x = Normal(mu=tf.ones(M) * mu, sigma=tf.ones(M))

      qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))

      x_ph = tf.placeholder(tf.float32, [M])
      data = {x: x_ph}
      inference = ed.KLqp({mu: qmu}, data)
      inference.initialize(scale={x: float(N) / M})
      self.assertAllEqual(inference.scale[x], float(N) / M)

  def test_scale_1d(self):
    with self.test_session():
      N = 10
      M = 5
      mu = Normal(mu=0.0, sigma=1.0)
      x = Normal(mu=tf.ones(M) * mu, sigma=tf.ones(M))

      qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))

      x_ph = tf.placeholder(tf.float32, [M])
      inference = ed.KLqp({mu: qmu}, data={x: x_ph})
      inference.initialize(scale={x: tf.range(M, dtype=tf.float32)})
      self.assertAllEqual(inference.scale[x].eval(), np.arange(M))

if __name__ == '__main__':
  tf.test.main()
