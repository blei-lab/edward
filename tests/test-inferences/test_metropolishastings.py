from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Empirical


class test_metropolishastings_class(tf.test.TestCase):

  def test_normalnormal_run(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones(50) * mu, scale=1.0)

      qmu = Empirical(params=tf.Variable(tf.ones(2000)))
      proposal_mu = Normal(loc=0.0, scale=1.0)

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.MetropolisHastings({mu: qmu},
                                        {mu: proposal_mu},
                                        data={x: x_data})
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1e-2)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
