from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_klpq_class(tf.test.TestCase):

  def test_normalnormal_run(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(mu=0.0, sigma=1.0)
      x = Normal(mu=tf.ones(50) * mu, sigma=1.0)

      qmu_mu = tf.Variable(tf.random_normal([]))
      qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))
      qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)

      # analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
      inference = ed.KLpq({mu: qmu}, data={x: x_data})
      inference.run(n_samples=25, n_iter=100)

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-1, atol=1e-1)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-1, atol=1e-1)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
