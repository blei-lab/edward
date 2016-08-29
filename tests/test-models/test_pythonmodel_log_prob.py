from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import PythonModel
from scipy.stats import beta, bernoulli


class BetaBernoulli(PythonModel):
  """p(x, p) = Bernoulli(x | p) * Beta(p | 1, 1)"""
  def _py_log_prob(self, xs, zs):
    # This example is written for pedagogy. We recommend
    # vectorizing operations in practice.
    xs = xs['x']
    ps = zs['p']
    n_samples = ps.shape[0]
    lp = np.zeros(n_samples, dtype=np.float32)
    for b in range(n_samples):
      lp[b] = beta.logpdf(ps[b, :], a=1.0, b=1.0)
      for n in range(xs.shape[0]):
        lp[b] += bernoulli.logpmf(xs[n], p=ps[b, :])

    return lp


def _test(model, xs, zs):
  n_samples = zs['p'].shape[0]
  val_true = np.zeros(n_samples, dtype=np.float32)
  for s in range(n_samples):
    p = np.squeeze(zs['p'][s, :])
    val_true[s] = beta.logpdf(p, 1, 1)
    val_true[s] += np.sum([bernoulli.logpmf(x, p)
                           for x in xs['x']])

  val_ed = model.log_prob(xs, zs)
  assert np.allclose(val_ed.eval(), val_true)
  zs_tf = {key: tf.cast(value, dtype=tf.float32)
           for key, value in six.iteritems(zs)}
  val_ed = model.log_prob(xs, zs_tf)
  assert np.allclose(val_ed.eval(), val_true)


class test_pythonmodel_log_prob_class(tf.test.TestCase):

  def test_1latent(self):
    with self.test_session():
      model = BetaBernoulli()
      data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
      zs = {'p': np.array([[0.5]])}
      _test(model, data, zs)
      zs = {'p': np.array([[0.4], [0.2], [0.2351], [0.6213]])}
      _test(model, data, zs)
