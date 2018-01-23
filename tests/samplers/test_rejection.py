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
      sampler = GammaRejectionSampler
      z = sampler.h(epsilon, alpha, beta)

      self.assertAllClose(sampler.h_inverse(z, alpha, beta).eval(),
        epsilon.eval(), atol=1e-6)
      # np.log(scipy.stats.norm(.5))
      self.assertAllClose(sampler.log_prob_s(epsilon).eval(),
        -1.0439385332046727, atol=1e-6)
