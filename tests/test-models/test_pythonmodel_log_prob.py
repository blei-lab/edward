from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import PythonModel
from scipy.stats import beta, bernoulli


class BetaBernoulli(PythonModel):
  """p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)"""
  def _py_log_prob(self, xs, zs):
    # This example is written for pedagogy. We recommend
    # vectorizing operations in practice.
    n_minibatch = zs.shape[0]
    lp = np.zeros(n_minibatch, dtype=np.float32)
    for b in range(n_minibatch):
      lp[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
      for n in range(xs['x'].shape[0]):
        lp[b] += bernoulli.logpmf(xs['x'][n], p=zs[b, :])

    return lp

def _test(model, xs, zs):
  n_samples = zs.shape[0]
  val_true = np.zeros(n_samples, dtype=np.float32)
  for s in range(n_samples):
    p = np.squeeze(zs[s, :])
    val_true[s] = beta.logpdf(p, 1, 1)
    val_true[s] += np.sum([bernoulli.logpmf(x, p)
                 for x in xs['x']])

  val_ed = model.log_prob(xs, zs)
  assert np.allclose(val_ed.eval(), val_true)
  zs_tf = tf.cast(zs, dtype=tf.float32)
  val_ed = model.log_prob(xs, zs_tf)
  assert np.allclose(val_ed.eval(), val_true)

class test_pythonmodel_log_prob_class(tf.test.TestCase):

  def test_1latent(self):
    with self.test_session():
      model = BetaBernoulli()
      data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
      zs = np.array([[0.5]])
      _test(model, data, zs)
      zs = np.array([[0.4], [0.2], [0.2351], [0.6213]])
      _test(model, data, zs)
