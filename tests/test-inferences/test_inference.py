from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm


class NormalNormal:
  """p(x, mu) = Normal(x | mu, 1) Normal(mu | 1, 1)"""
  def log_prob(self, xs, zs):
    log_prior = norm.logpdf(zs['mu'], 1.0, 1.0)
    log_lik = tf.reduce_sum(norm.logpdf(xs['x'], zs['mu'], 1.0))
    return log_lik + log_prior


class test_inference_class(tf.test.TestCase):

  def test_latent_vars(self):
    tf.InteractiveSession()
    mu = Normal(mu=0.0, sigma=1.0)
    qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))
    qmu_misshape = Normal(mu=tf.constant([0.0]), sigma=tf.constant([1.0]))

    ed.Inference({mu: qmu})
    self.assertRaises(TypeError, ed.Inference, {mu: '5'})
    self.assertRaises(TypeError, ed.Inference, {mu: tf.constant(0.0)})
    self.assertRaises(TypeError, ed.Inference, {tf.constant(0.0): qmu})
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
    self.assertRaises(TypeError, ed.Inference, data={5: tf.constant(0.0)})
    self.assertRaises(TypeError, ed.Inference, data={x: tf.zeros(5)})
    self.assertRaises(TypeError, ed.Inference, data={x_ph: x})
    self.assertRaises(TypeError, ed.Inference, data={x: qx_misshape})

  def test_model_wrapper(self):
    tf.InteractiveSession()
    model = NormalNormal()
    qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))

    ed.Inference({'mu': qmu}, model_wrapper=model)

if __name__ == '__main__':
  tf.test.main()
