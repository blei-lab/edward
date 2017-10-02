from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, InverseGamma
from tensorflow.contrib.distributions import bijectors


class test_marginal_class(tf.test.TestCase):

  def test_bad_graph(self):
    with self.test_session():
      loc = Normal(tf.zeros(5), 5.0)
      y_loc = tf.expand_dims(loc, 1)  # this displaces the sample dimension
      inv_scale = Normal(tf.zeros(3), 1.0)
      y_scale = tf.expand_dims(tf.nn.softplus(inv_scale), 0)
      y = Normal(y_loc, y_scale)
      with self.assertRaises(ValueError):
        ed.marginal(y, 20)

  def test_sample_arg(self):
    with self.test_session():
      y = Normal(0.0, 1.0, sample_shape=10)
      with self.assertRaises(NotImplementedError):
        ed.marginal(y, 20)

  def test_sample_arg_ancestor(self):
    with self.test_session():
      x = Normal(0.0, 1.0, sample_shape=10)
      y = Normal(x, 0.0)
      with self.assertRaises(NotImplementedError):
        ed.marginal(y, 20)

  def test_no_ancestor(self):
    with self.test_session():
      y = Normal(0.0, 1.0)
      sample = ed.marginal(y, 4)
      self.assertEqual(sample.shape, [4])

  def test_no_ancestor_batch(self):
    with self.test_session():
      y = Normal(tf.zeros([2, 3, 4]), 1.0)
      sample = ed.marginal(y, 5)
      self.assertEqual(sample.shape, [5, 2, 3, 4])

  def test_single_ancestor(self):
    with self.test_session():
      loc = Normal(0.0, 1.0)
      y = Normal(loc, 1.0)
      sample = ed.marginal(y, 4)
      self.assertEqual(sample.shape, [4])

  def test_single_ancestor_batch(self):
    with self.test_session():
      loc = Normal(tf.zeros([2, 3, 4]), 1.0)
      y = Normal(loc, 1.0)
      sample = ed.marginal(y, 5)
      self.assertEqual(sample.shape, [5, 2, 3, 4])

  def test_sample_passthrough(self):
    with self.test_session():
      loc = Normal(0.0, 100.0)
      y = Normal(loc, 0.0001)
      conditional_sample = y.sample(50)
      marginal_sample = ed.marginal(y, 50)
      self.assertTrue(np.std(conditional_sample.eval()) < 1.0)
      self.assertTrue(np.std(marginal_sample.eval()) > 1.0)

  def test_multiple_ancestors(self):
    with self.test_session():
      loc = Normal(0.0, 1.0)
      scale = InverseGamma(1.0, 1.0)
      y = Normal(loc, scale)
      sample = ed.marginal(y, 4)
      self.assertEqual(sample.shape, [4])

  def test_multiple_ancestors_batch(self):
    with self.test_session():
      loc = Normal(tf.zeros(5), 1.0)
      scale = InverseGamma(tf.ones(5), 1.0)
      y = Normal(loc, scale)
      sample = ed.marginal(y, 4)
      self.assertEqual(sample.shape, [4, 5])

  def test_multiple_ancestors_batch_broadcast(self):
    with self.test_session():
      loc = Normal(tf.zeros([5, 1]), 1.0)
      scale = InverseGamma(tf.ones([1, 6]), 1.0)
      y = Normal(loc, scale)
      sample = ed.marginal(y, 4)
      self.assertEqual(sample.shape, [4, 5, 6])

  def test_multiple_ancestors_failed_broadcast(self):
    with self.test_session():
      loc = Normal(tf.zeros([5, 1]), 1.0)
      scale = InverseGamma(tf.ones([6]), 1.0)
      y = Normal(loc, scale)
      with self.assertRaises(ValueError):
        sample = ed.marginal(y, 4)
