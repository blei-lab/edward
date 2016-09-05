from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import pymc3 as pm
import six
import tensorflow as tf
import theano

from edward.models import PyMC3Model
from scipy.stats import bernoulli, beta


def _test(model, xs, zs):
  val_true = beta.logpdf(zs['p'], 1.0, 1.0)
  val_true += np.sum([bernoulli.logpmf(x, zs['p'])
                      for x in list(six.itervalues(xs))[0]])
  val_ed = model.log_prob(xs, zs)
  assert np.allclose(val_ed.eval(), val_true)
  zs_tf = {key: tf.cast(value, dtype=tf.float32)
           for key, value in six.iteritems(zs)}
  val_ed = model.log_prob(xs, zs_tf)
  assert np.allclose(val_ed.eval(), val_true)


class test_pymc3_log_prob_class(tf.test.TestCase):

  def test_1latent(self):
    with self.test_session():
      x_obs = theano.shared(np.zeros(1))
      with pm.Model() as pm_model:
        p = pm.Beta('p', 1, 1, transform=None)
        x = pm.Bernoulli('x', p, observed=x_obs)

      model = PyMC3Model(pm_model)
      data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}
      zs = {'p': np.array(0.5)}
      _test(model, data, zs)

if __name__ == '__main__':
  tf.test.main()
