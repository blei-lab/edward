from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Empirical


class test_sghmc_class(tf.test.TestCase):

  def test_normalnormal_float32(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      qmu = Empirical(params=tf.Variable(tf.ones(5000)))

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.SGHMC({mu: qmu}, data={x: x_data})
      inference.run(step_size=0.025)

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1.5e-2)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=5e-2, atol=5e-2)

  def test_normalnormal_float64(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float64)

      mu = Normal(loc=tf.constant(0.0, dtype=tf.float64),
                  scale=tf.constant(1.0, dtype=tf.float64))
      x = Normal(loc=mu,
                 scale=tf.constant(1.0, dtype=tf.float64),
                 sample_shape=50)

      qmu = Empirical(params=tf.Variable(tf.ones(5000, dtype=tf.float64)))

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.SGHMC({mu: qmu}, data={x: x_data})
      inference.run(step_size=0.025)

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1.5e-2)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=5e-2, atol=5e-2)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
