from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Beta, Dirichlet, Normal, ParamMixture


class test_param_mixture_class(tf.test.TestCase):

  def _test_shape(self, *args, **kwargs):
    g = tf.Graph()
    with g.as_default():
      x = ParamMixture(*args, **kwargs)
      val_est = x.shape
      val_true = x.cat.get_sample_shape().concatenate(
          x.cat.get_batch_shape()).concatenate(x.components.get_event_shape())
      self.assertEqual(val_est, val_true)

  def _test_stats(self, *args, **kwargs):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(10003)

      x = ParamMixture(*args, **kwargs)
      cat = x.cat
      components = x.components
      comp_means = components.mean()
      comp_stddevs = components.std()
      marginal_mean = x.mean()
      marginal_stddev = x.std()
      marginal_var = x.variance()

    with self.test_session(graph=g) as sess:
      to_eval = [x, cat, comp_means, comp_stddevs, marginal_mean,
                 marginal_stddev, marginal_var]
      vals = sess.run(to_eval)
      vals = {k: v for k, v in zip(to_eval, vals)}

    self.assertAllClose(vals[x].mean(0), vals[marginal_mean],
                        rtol=0.01, atol=0.01)
    self.assertAllClose(vals[x].std(0), vals[marginal_stddev],
                        rtol=0.01, atol=0.01)
    self.assertAllClose(vals[x].var(0), vals[marginal_var],
                        rtol=0.01, atol=0.01)
    for k in range(x.num_components):
      selector = (vals[cat] == k)
      self.assertAllClose(selector.mean(), pi[k], rtol=0.01, atol=0.01)
      x_k = vals[x][selector]
      self.assertAllClose(x_k.mean(0), vals[comp_means][k],
                          rtol=0.05, atol=0.05)
      self.assertAllClose(x_k.std(0), vals[comp_stddevs][k],
                          rtol=0.05, atol=0.05)

  def test_normal_0d(self):
    pi = np.array([0.2, 0.3, 0.5], np.float32)
    mu = np.array([1.0, 5.0, 7.0], np.float32)
    sigma = np.array([1.5, 1.5, 1.5], np.float32)

    self._test_shape(pi, {'mu': mu, 'sigma': sigma}, Normal)
    self._test_stats(pi, {'mu': mu, 'sigma': sigma}, Normal, sample_shape=10000)

  def test_beta_0d(self):
    pi = np.array([0.2, 0.3, 0.5], np.float32)
    a = np.array([2.0, 1.0, 0.5], np.float32)
    b = a + 2.0

    self._test_shape(pi, {'a': a, 'b': b}, Beta)
    self._test_stats(pi, {'a': a, 'b': b}, Beta, sample_shape=10000)

  def test_beta_1d(self):
    pi_broadcast = np.array([0.2, 0.3, 0.5], np.float32)
    pi = np.tile(pi_broadcast, [2, 1])
    a = np.array([[2.0, 0.5], [1.0, 1.0], [0.5, 2.0]], np.float32)
    b = a + 2.0

    # self._test_shape(pi_broadcast, {'a': a, 'b': b}, Beta)
    self._test_shape(pi, {'a': a, 'b': b}, Beta)
    self._test_stats(pi, {'a': a, 'b': b}, Beta, sample_shape=10000)

  # def test_dirichlet_1d(self):
  #   # TODO
  #   pi = np.array([0.4, 0.6], np.float32)
  #   alpha = np.ones([2, 3], np.float32).T

  #   self._test(pi, {'alpha': alpha}, Dirichlet)

if __name__ == '__main__':
  tf.test.main()
