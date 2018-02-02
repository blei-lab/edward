from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Beta, Normal, ParamMixture


def _make_histograms(values, hists, hist_centers, x_axis, n_bins):
  if len(values.shape) > 1:
    for i in range(values.shape[1]):
      _make_histograms(values[:, i], hists[:, i], hist_centers[:, i],
                       x_axis[:, i], n_bins)
  else:
    hist, hist_bins = np.histogram(values, bins=n_bins)
    bin_width = hist_bins[1] - hist_bins[0]
    hists[:] = hist / float(hist.sum())
    hist_centers[:] = 0.5 * (hist_bins[1:] + hist_bins[:-1])
    x_axis[:n_bins] = hist_centers


class test_param_mixture_class(tf.test.TestCase):

  def _test(self, probs, params, dist):
    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(10003)

      N = 50000

      x = ParamMixture(probs, params, dist, sample_shape=N)
      cat = x.cat
      components = x.components

      marginal_logp = x.marginal_log_prob(x)
      cond_logp = x.log_prob(x)

      comp_means = components.mean()
      comp_stddevs = components.stddev()
      marginal_mean = x.mean()
      marginal_stddev = x.stddev()
      marginal_var = x.variance()

    sess = self.test_session(graph=g)
    with self.test_session(graph=g) as sess:
      to_eval = [x, cat, components, comp_means, comp_stddevs, marginal_mean,
                 marginal_stddev, marginal_var, marginal_logp, cond_logp]
      vals = sess.run(to_eval)
      vals = {k: v for k, v in zip(to_eval, vals)}

      # Test that marginal statistics are reasonable
      self.assertAllClose(vals[x].mean(0), vals[marginal_mean],
                          rtol=0.01, atol=0.01)
      self.assertAllClose(vals[x].std(0), vals[marginal_stddev],
                          rtol=0.01, atol=0.01)
      self.assertAllClose(vals[x].var(0), vals[marginal_var],
                          rtol=0.01, atol=0.01)

      # Test that per-component statistics are reasonable
      for k in range(x.num_components):
        selector = (vals[cat] == k)
        self.assertAllClose(selector.mean(), probs[k], rtol=0.01, atol=0.01)
        x_k = vals[x][selector]
        self.assertAllClose(x_k.mean(0), vals[comp_means][k],
                            rtol=0.05, atol=0.05)
        self.assertAllClose(x_k.std(0), vals[comp_stddevs][k],
                            rtol=0.05, atol=0.05)

      n_bins = 100
      x_hists = np.zeros((n_bins,) + vals[x].shape[1:])
      hist_centers = np.zeros_like(x_hists)
      x_axis = np.zeros((N,) + vals[x].shape[1:])
      _make_histograms(vals[x], x_hists, hist_centers, x_axis, n_bins)

      x_marginal_val = sess.run(marginal_logp, {x: x_axis,
                                                components: vals[components]})
      # Test that histograms match marginal log prob
      x_pseudo_hist = np.exp(x_marginal_val[:n_bins])
      self.assertAllClose(x_pseudo_hist.sum(0) * (x_axis[1] - x_axis[0]), 1.,
                          rtol=0.1, atol=0.1)
      x_pseudo_hist /= x_pseudo_hist.sum(0, keepdims=True)
      self.assertLess(abs(x_pseudo_hist - x_hists).sum(0).mean(), 0.1)

      # Test that histograms match conditional log prob
      for k in range(probs.shape[-1]):
        k_cat = k + np.zeros(x_axis.shape, np.int32)
        x_vals_k = sess.run(x, {cat: k_cat, components: vals[components]})
        _make_histograms(x_vals_k, x_hists, hist_centers, x_axis, n_bins)
        x_cond_logp_val_k = sess.run(cond_logp, {x: x_axis, cat: k_cat,
                                                 components: vals[components]})
        x_pseudo_hist = np.exp(x_cond_logp_val_k[:n_bins])
        self.assertAllClose(x_pseudo_hist.sum(0) * (x_axis[1] - x_axis[0]), 1.,
                            rtol=0.1, atol=0.1)
        x_pseudo_hist /= x_pseudo_hist.sum(0, keepdims=True)
        self.assertLess(abs(x_pseudo_hist - x_hists).sum(0).mean(), 0.1)

  def test_normal(self):
    """Mixture of 3 normal distributions."""
    probs = np.array([0.2, 0.3, 0.5], np.float32)
    loc = np.array([1.0, 5.0, 7.0], np.float32)
    scale = np.array([1.5, 1.5, 1.5], np.float32)

    self._test(probs, {'loc': loc, 'scale': scale}, Normal)

  def test_beta(self):
    """Mixture of 3 beta distributions."""
    probs = np.array([0.2, 0.3, 0.5], np.float32)
    conc1 = np.array([2.0, 1.0, 0.5], np.float32)
    conc0 = conc1 + 2.0

    self._test(probs, {'concentration1': conc1, 'concentration0': conc0},
               Beta)

  def test_batch_beta(self):
    """Two mixtures of 3 beta distributions."""
    probs = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], np.float32)
    conc1 = np.array([[2.0, 0.5], [1.0, 1.0], [0.5, 2.0]], np.float32)
    conc0 = conc1 + 2.0

    # self._test(probs, {'concentration1': conc1, 'concentration0': conc0},
    #            Beta)
    self.assertRaises(NotImplementedError,
                      self._test, probs,
                      {'concentration1': conc1, 'concentration0': conc0},
                      Beta)

if __name__ == '__main__':
  tf.test.main()
