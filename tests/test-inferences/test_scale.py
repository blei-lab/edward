from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_inference_scale_class(tf.test.TestCase):

  def test_scale_0d(self):
    N = 10
    M = 5
    mu = Normal(mu=0.0, sigma=1.0)
    x = Normal(mu=tf.ones(M) * mu, sigma=tf.ones(M))

    qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))

    x_ph = tf.placeholder(tf.float32, [M])
    data = {x: x_ph}
    inference = ed.KLqp({mu: qmu}, data)
    inference.initialize(scale={x: float(N) / M})
    assert inference.scale[x] == float(N) / M

  def test_scale_1d(self):
    N = 10
    M = 5
    mu = Normal(mu=0.0, sigma=1.0)
    x = Normal(mu=tf.ones(M) * mu, sigma=tf.ones(M))

    qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))

    x_ph = tf.placeholder(tf.float32, [M])
    data = {x: x_ph}
    inference = ed.KLqp({mu: qmu}, data)
    inference.initialize(scale={x: tf.cast(tf.range(M), tf.float32)})
    with self.test_session():
      self.assertAllClose(inference.scale[x].eval(), np.arange(M))

if __name__ == '__main__':
  tf.test.main()
