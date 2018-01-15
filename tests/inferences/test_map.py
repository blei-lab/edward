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

  def test_normalnormal_regularization(self):
    with self.test_session() as sess:
      x_data = np.array([5.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      qmu = PointMass(params=tf.Variable(1.0))

      inference = ed.MAP({mu: qmu}, data={x: x_data})
      inference.run(n_iter=1000)
      mu_val = qmu.mean().eval()

      # regularized solution
      regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
      mu_reg = tf.get_variable("mu_reg", shape=[],
                               regularizer=regularizer)
      x_reg = Normal(loc=mu_reg, scale=1.0, sample_shape=50)

      inference_reg = ed.MAP(None, data={x_reg: x_data})
      inference_reg.run(n_iter=1000)

      mu_reg_val = mu_reg.eval()
      self.assertAllClose(mu_val, mu_reg_val)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
