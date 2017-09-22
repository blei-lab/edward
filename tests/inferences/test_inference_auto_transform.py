from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Gamma, Normal


class test_inference_auto_transform_class(tf.test.TestCase):

  def test_auto_transform_true(self):
    with self.test_session():
      x = Gamma(2.0, 2.0)
      qx = Normal(loc=tf.Variable(tf.random_normal([])),
                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

      inference = ed.KLqp({x: qx})
      inference.initialize(auto_transform=True, n_samples=5, n_iter=150)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      self.assertAllClose(info_dict['loss'], 0.0, rtol=0.2, atol=0.2)

  def test_auto_transform_false(self):
    with self.test_session():
      x = Gamma(2.0, 2.0)
      qx = Normal(loc=tf.Variable(tf.random_normal([])),
                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

      inference = ed.KLqp({x: qx})
      inference.initialize(auto_transform=False, n_samples=5, n_iter=150)
      tf.global_variables_initializer().run()
      for _ in range(inference.n_iter):
        info_dict = inference.update()

      self.assertAllEqual(info_dict['loss'], np.nan)

if __name__ == '__main__':
  ed.set_seed(124125)
  tf.test.main()
