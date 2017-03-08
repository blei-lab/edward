from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_inference_class(tf.test.TestCase):

  def test_latent_vars(self):
    tf.InteractiveSession()
    mu = Normal(mu=0.0, sigma=1.0)
    qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))
    qmu_misshape = Normal(mu=tf.constant([0.0]), sigma=tf.constant([1.0]))

    ed.Inference({mu: qmu})
    ed.Inference({mu: tf.constant(0.0)})
    ed.Inference({tf.constant(0.0): qmu})
    self.assertRaises(TypeError, ed.Inference, {mu: '5'})
    self.assertRaises(TypeError, ed.Inference, {mu: qmu_misshape})

  def test_data(self):
    tf.InteractiveSession()
    x = Normal(mu=0.0, sigma=1.0)
    qx = Normal(mu=0.0, sigma=1.0)
    qx_misshape = Normal(mu=tf.constant([0.0]), sigma=tf.constant([1.0]))
    x_ph = tf.placeholder(tf.float32)

    ed.Inference()
    ed.Inference(data={x: tf.constant(0.0)})
    ed.Inference(data={x_ph: tf.constant(0.0)})
    ed.Inference(data={x: np.float64(0.0)})
    ed.Inference(data={x: np.int64(0)})
    ed.Inference(data={x: 0.0})
    ed.Inference(data={x: 0})
    ed.Inference(data={x: False})  # converted to `int`
    ed.Inference(data={x: x_ph})
    ed.Inference(data={x: qx})
    ed.Inference(data={2.0 * x: tf.constant(0.0)})
    self.assertRaises(TypeError, ed.Inference, data={5: tf.constant(0.0)})
    self.assertRaises(TypeError, ed.Inference, data={x: tf.zeros(5)})
    self.assertRaises(TypeError, ed.Inference, data={x_ph: x})
    self.assertRaises(TypeError, ed.Inference, data={x: qx_misshape})

if __name__ == '__main__':
  tf.test.main()
