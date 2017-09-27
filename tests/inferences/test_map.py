from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, PointMass


class test_map_class(tf.test.TestCase):

  def test_normalnormal_run(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      qmu = PointMass(params=tf.Variable(1.0))

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.MAP({mu: qmu}, data={x: x_data})
      inference.run(n_iter=1000)

      self.assertAllClose(qmu.mean().eval(), 0)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
