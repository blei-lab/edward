from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Empirical, Normal


class test_hmc_class(tf.test.TestCase):

  def test_normalnormal_float32(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      qmu = Empirical(params=tf.Variable(tf.ones(2000)))

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.HMC({mu: qmu}, data={x: x_data})
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1e-2)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-2, atol=1e-2)

  def test_normalnormal_float64(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float64)

      mu = Normal(loc=tf.constant(0.0, dtype=tf.float64),
                  scale=tf.constant(1.0, dtype=tf.float64))
      x = Normal(loc=mu,
                 scale=tf.constant(1.0, dtype=tf.float64),
                 sample_shape=50)

      qmu = Empirical(params=tf.Variable(tf.ones(2000, dtype=tf.float64)))

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.HMC({mu: qmu}, data={x: x_data})
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1e-2)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-2, atol=1e-2)

  def test_indexedslices(self):
    """Test that gradients accumulate when tf.gradients doesn't return
    tf.Tensor (IndexedSlices)."""
    with self.test_session() as sess:
      N = 10  # number of data points
      K = 2  # number of clusters
      T = 1  # number of MCMC samples

      x_data = np.zeros(N, dtype=np.float32)

      mu = Normal(0.0, 1.0, sample_shape=K)
      c = Categorical(logits=tf.zeros(N))
      x = Normal(tf.gather(mu, c), tf.ones(N))

      qmu = Empirical(params=tf.Variable(tf.ones([T, K])))
      qc = Empirical(params=tf.Variable(tf.ones([T, N])))

      inference = ed.HMC({mu: qmu}, data={x: x_data})
      inference.initialize()

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
