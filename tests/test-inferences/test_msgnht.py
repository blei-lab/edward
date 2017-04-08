from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Empirical


class test_sgld_class(tf.test.TestCase):

  def test_normalnormal_run(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(mu=0.0, sigma=1.0)
      x = Normal(mu=tf.ones(50) * mu, sigma=1.0)

      qmu = Empirical(params=tf.Variable(tf.ones(4000)))

      # analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
      # Since, in this case, unknown noise inserted by stochastic gradients does not exists,
      # D is supposed to be 0
      inference = ed.mSGNHT({mu: qmu}, data={x: x_data})
      inference.run(step_size=0.00005)

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1e-2)
      self.assertAllClose(qmu.std().eval(), np.sqrt(1 / 51),
                          rtol=5e-2, atol=5e-2)
      v = qmu.get_variables()[0].eval()
      for i in range(5000):
        print(v[i])

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
