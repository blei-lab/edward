from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Normal
from edward.util import check_latent_vars


class test_check_latent_vars_class(tf.test.TestCase):

  def test(self):
    with self.test_session():
      mu = Normal(0.0, 1.0)
      qmu = Normal(tf.Variable(0.0), tf.constant(1.0))
      qmu_vec = Normal(tf.constant([0.0]), tf.constant([1.0]))

      check_latent_vars({mu: qmu})
      check_latent_vars({mu: tf.constant(0.0)})
      check_latent_vars({tf.constant(0.0): qmu})
      self.assertRaises(TypeError, check_latent_vars, {mu: '5'})
      self.assertRaises(TypeError, check_latent_vars, {mu: qmu_vec})

if __name__ == '__main__':
  tf.test.main()
