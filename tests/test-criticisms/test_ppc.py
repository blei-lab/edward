from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal


class test_ppc_class(tf.test.TestCase):

  def test_data(self):
    with self.test_session():
      x = Normal(loc=0.0, scale=1.0)
      y = 2.0 * x
      x_data = tf.constant(0.0)
      y_data = tf.constant(0.0)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x]), {x: x_data}, n_samples=1)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[y]), {y: y_data}, n_samples=1)
      self.assertRaises(TypeError, ed.ppc, lambda xs, zs: tf.reduce_mean(xs[y]),
                        {'y': y_data}, n_samples=1)

  def test_latent_vars(self):
    with self.test_session():
      x = Normal(loc=0.0, scale=1.0)
      y = 2.0 * x
      z = Normal(loc=0.0, scale=1.0)
      x_data = tf.constant(0.0)
      y_data = tf.constant(0.0)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x]) + tf.reduce_mean(zs[z]),
             {x: x_data}, {z: z}, n_samples=1)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x]) + tf.reduce_mean(zs[z]),
             {x: x_data}, {z: y}, n_samples=1)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x]) + tf.reduce_mean(zs[y]),
             {x: x_data}, {y: y}, n_samples=1)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x]) + tf.reduce_mean(zs[y]),
             {x: x_data}, {y: z}, n_samples=1)
      self.assertRaises(TypeError, ed.ppc, lambda xs, zs: tf.reduce_mean(xs[x]),
                        {x: x_data}, {'y': z}, n_samples=1)

  def test_n_samples(self):
    with self.test_session():
      x = Normal(loc=0.0, scale=1.0)
      x_data = tf.constant(0.0)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x]), {x: x_data}, n_samples=1)
      ed.ppc(lambda xs, zs: tf.reduce_mean(xs[x]), {x: x_data}, n_samples=5)
      self.assertRaises(TypeError, ed.ppc, lambda xs, zs: tf.reduce_mean(xs[x]),
                        {x: x_data}, n_samples='1')

if __name__ == '__main__':
  tf.test.main()
