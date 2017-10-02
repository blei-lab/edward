from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from tensorflow.contrib.distributions import bijectors


class test_marginal_class(tf.test.TestCase):

  def test_bad_graph(self):
    with self.test_session():
      loc = Normal(0.0, 5.0, sample_shape=5)
      y_loc = tf.expand_dims(loc, 1)
      inv_scale = Normal(0.0, 1.0, sample_shape=3)
      y_scale = tf.expand_dims(tf.nn.softplus(inv_scale), 0)
      y = Normal(y_loc, y_scale)
      with self.assertRaises(ValueError):
        ed.marginal(y, 20)

  def test_single(self):
    with self.test_session():
      y = Normal(0.0, 1.0)
      print(ed.get_ancestors(y))
      sample = ed.marginal(y, 4)
      self.assertEqual(sample.shape, [4])

  def test_single_expand(self):
    with self.test_session():
      y = Normal(0.0, 1.0, sample_shape=5)
      print(ed.get_ancestors(y))
      sample = ed.marginal(y, 4)
      self.assertEqual(sample.shape, [4, 5])
