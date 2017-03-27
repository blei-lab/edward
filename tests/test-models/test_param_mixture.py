from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import ParamMixture, Normal, Beta


class test_param_mixture_class(tf.test.TestCase):

  def _test(self, pi, params, dist):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(10003)

      N = 10000
      K = len(pi)

      x = ParamMixture(pi, params, dist, sample_shape=N)
      z = x.cat
      components = x.components
      comp_means = components.mean()
      comp_stddevs = components.std()
      marginal_mean = x.mean()
      marginal_stddev = x.std()
      marginal_var = x.variance()

    with self.test_session(graph=g):
      sess = tf.get_default_session()
      to_eval = [x, z, comp_means, comp_stddevs, marginal_mean,
                 marginal_stddev, marginal_var]
      vals = sess.run(to_eval)
      vals = {to_eval[i]: vals[i] for i in range(len(vals))}

    self.assertEqual(vals[z].shape, (N,))
    self.assertAllClose(vals[x].mean(0), vals[marginal_mean],
                        rtol=0.01, atol=0.01)
    self.assertAllClose(vals[x].std(0), vals[marginal_stddev],
                        rtol=0.01, atol=0.01)
    self.assertAllClose(vals[x].var(0), vals[marginal_var],
                        rtol=0.01, atol=0.01)
    for k in range(K):
      self.assertAllClose((vals[z] == k).mean(), pi[k], rtol=0.01, atol=0.01)
      x_k = vals[x][vals[z] == k]
      self.assertAllClose(x_k.mean(0), vals[comp_means][k],
                          rtol=0.05, atol=0.05)
      self.assertAllClose(x_k.std(0), vals[comp_stddevs][k],
                          rtol=0.05, atol=0.05)

  def test_mog(self):
    pi = np.array([0.2, 0.3, 0.5], np.float32)
    mu = np.array([1.0, 5.0, 7.0], np.float32)
    sigma = 1.5

    self._test(pi, {'mu': mu, 'sigma': sigma}, Normal)

  def test_mob(self):
    pi = np.array([0.2, 0.3, 0.5], np.float32)
    a = np.array([2.0, 1.0, 0.5], np.float32)
    b = np.array([2.0, 1.0, 0.5], np.float32) + 2.

    self._test(pi, {'a': a, 'b': b}, Beta)

  def test_batched_mob(self):
    pi = np.array([0.2, 0.3, 0.5], np.float32)
    a = np.array([[2.0, 1.0, 0.5], [0.5, 1.0, 2.0]], np.float32).T
    b = np.array([[2.0, 1.0, 0.5], [0.5, 1.0, 2.0]], np.float32).T + 2.0

    self._test(pi, {'a': a, 'b': b}, Beta)

if __name__ == '__main__':
  tf.test.main()
