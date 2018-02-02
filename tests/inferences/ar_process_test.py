from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, PointMass
from scipy.optimize import minimize

from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution
from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED


class AutoRegressive(RandomVariable, Distribution):
  # a 1-D AR(1) process
  # a[t + 1] = a[t] + eps with eps ~ N(0, sig**2)
  def __init__(self, T, a, sig, *args, **kwargs):
    self.a = a
    self.sig = sig
    self.T = T
    self.shocks = Normal(tf.zeros(T), scale=sig)
    self.z = tf.scan(lambda acc, x: self.a * acc + x, self.shocks)

    if 'dtype' not in kwargs:
      kwargs['dtype'] = tf.float32
    if 'allow_nan_stats' not in kwargs:
      kwargs['allow_nan_stats'] = False
    if 'reparameterization_type' not in kwargs:
      kwargs['reparameterization_type'] = FULLY_REPARAMETERIZED
    if 'validate_args' not in kwargs:
      kwargs['validate_args'] = False
    if 'name' not in kwargs:
      kwargs['name'] = 'AutoRegressive'

    super(AutoRegressive, self).__init__(*args, **kwargs)

    self._args = (T, a, sig)

  def _log_prob(self, value):
    err = value - self.a * tf.pad(value[:-1], [[1, 0]], 'CONSTANT')
    lpdf = self.shocks._log_prob(err)
    return tf.reduce_sum(lpdf)

  def _sample_n(self, n, seed=None):
    return tf.scan(lambda acc, x: self.a * acc + x,
                   self.shocks._sample_n(n, seed))


class test_ar_process(tf.test.TestCase):

  def test_ar_mle(self):
    # set up test data: a random walk
    T = 100
    z_true = np.zeros(T)
    r = 0.95
    sig = 0.01
    eta = 0.01
    for t in range(1, 100):
      z_true[t] = r * z_true[t - 1] + sig * np.random.randn()

    x_data = (z_true + eta * np.random.randn(T)).astype(np.float32)

    # use scipy to find max likelihood
    def cost(z):
      initial = z[0]**2 / sig**2
      ar = np.sum((z[1:] - r * z[:-1])**2) / sig**2
      data = np.sum((x_data - z)**2) / eta**2
      return initial + ar + data

    mle = minimize(cost, np.zeros(T)).x

    with self.test_session() as sess:
      z = AutoRegressive(T, r, sig)
      x = Normal(loc=z, scale=eta)

      qz = PointMass(params=tf.Variable(tf.zeros(T)))
      inference = ed.MAP({z: qz}, data={x: x_data})
      inference.run(n_iter=500)

      self.assertAllClose(qz.eval(), mle, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
