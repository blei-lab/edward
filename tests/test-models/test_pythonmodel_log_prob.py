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
    log_prior = beta.logpdf(zs['p'], a=1.0, b=1.0)
    log_lik = np.sum(bernoulli.logpmf(xs['x'], p=zs['p']))
    return log_lik + log_prior


def _test(model, xs, zs):
  val_true = beta.logpdf(zs['p'], 1.0, 1.0)
  val_true += np.sum([bernoulli.logpmf(x, zs['p'])
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
      zs = {'p': np.array(0.5)}
      _test(model, data, zs)

if __name__ == '__main__':
  tf.test.main()
