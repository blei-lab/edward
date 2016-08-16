from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from scipy.stats import beta, bernoulli


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


class test_stanmodel_log_prob_class(tf.test.TestCase):

  def test_1latent(self):
    model_code = """
      data {
        int<lower=0> N;
        int<lower=0,upper=1> x[N];
      }
      parameters {
        real<lower=0,upper=1> p;
      }
      model {
        p ~ beta(1.0, 1.0);
        for (n in 1:N)
        x[n] ~ bernoulli(p);
      }
    """
    with self.test_session():
      model = ed.StanModel(model_code=model_code)
      data = {'N': 10, 'x': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1]}
      zs = np.array([[0.5]])
      _test(model, data, zs)
      zs = np.array([[0.4], [0.2], [0.2351], [0.6213]])
      _test(model, data, zs)
