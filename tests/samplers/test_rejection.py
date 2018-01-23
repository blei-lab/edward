from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.samplers import GammaRejectionSampler


class test_rejection_samplers_class(tf.test.TestCase):

  def test_gamma_rejection_sampler(self):
    alpha = tf.constant(4.)
    beta = tf.constant(2.)
    epsilon = tf.constant(.5)
    with self.test_session() as sess:
      z = GammaRejectionSampler.h(epsilon, alpha, beta)
      self.assertAllClose(GammaRejectionSampler.h_inverse(z, alpha, beta).eval(),
        epsilon.eval(), atol=1e-6)
