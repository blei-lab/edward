from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, PointMass


class test_map_loss_class(tf.test.TestCase):

  def test_normal_loss_run(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)
      print('Without regularization\n-------------------')
      mu = Normal(mu=0.0, sigma=1.0)
      x = Normal(mu=tf.ones(50) * mu, sigma=1.0)
      qmu = PointMass(params=tf.Variable(tf.ones([])))

      # analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
      inference = ed.MAP({mu: qmu}, data={x: x_data})
      inference.run(n_iter=1000)
      qmu_no_reg = qmu.eval()

      # Add regularization on mu
      print('\nWith regularization\n-------------------')
      weight_decay = 0.005
      mu = tf.get_variable('mu', shape=(), dtype=tf.float32,
                          initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
                          regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                          trainable=True)
      mu = Normal(mu=mu, sigma=1.0)
      x = Normal(mu=tf.ones(50) * mu, sigma=1.0)
      qmu = PointMass(params=tf.Variable(tf.ones([])))
      inference = ed.MAP({mu: qmu}, data={x: x_data})
      inference.run(n_iter=1000)
      qmu_reg = qmu.eval()

      self.assertAllClose(qmu_no_reg, qmu_reg)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
