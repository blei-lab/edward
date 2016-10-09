from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multivariate_normal_diag, multivariate_normal_full
from scipy import stats


class test_multivariate_normal_entropy_class(tf.test.TestCase):

  def test_1d(self):
    diag = [1.0, 1.0]
    mu = tf.constant([0.0, 0.0])
    with self.test_session():
      self.assertAllClose(
          multivariate_normal_diag.entropy(mu=mu, diag_stdev=diag).eval(),
          stats.multivariate_normal.entropy(cov=np.diag(diag)))

  def test_2d_diag(self):
    cm = [[1.0, 0.0], [0.0, 1.0]]
    mu = tf.constant([0.0, 0.0])
    with self.test_session():
      self.assertAllClose(
          multivariate_normal_full.entropy(mu=mu, sigma=cm).eval(),
          stats.multivariate_normal.entropy(cov=np.array(cm)))

  def test_2d_full(self):
    cm = [[1.0, 0.9], [0.9, 1.0]]
    mu = tf.constant([0.0, 0.0])
    with self.test_session():
      self.assertAllClose(
          multivariate_normal_full.entropy(mu=mu, sigma=cm).eval(),
          stats.multivariate_normal.entropy(cov=np.array(cm)))

if __name__ == '__main__':
  tf.test.main()
