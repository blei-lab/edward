from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical, Normal


class test_gibbs_class(tf.test.TestCase):

  def test_beta_bernoulli(self):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

      p = Beta(a=1.0, b=1.0)
      x = Bernoulli(p=p, sample_shape=10)

      qp = Empirical(tf.Variable(tf.zeros(1000)))
      inference = ed.Gibbs({p: qp}, data={x: x_data})
      inference.run()

      true_posterior = Beta(a=3.0, b=9.0)

      val_est, val_true = sess.run([qp.mean(), true_posterior.mean()])
      self.assertAllClose(val_est, val_true, rtol=1e-2, atol=1e-2)

      val_est, val_true = sess.run([qp.variance(), true_posterior.variance()])
      self.assertAllClose(val_est, val_true, rtol=1e-2, atol=1e-2)

  def test_normal_normal(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(mu=0.0, sigma=1.0)
      x = Normal(mu=mu, sigma=1.0, sample_shape=50)

      qmu = Empirical(params=tf.Variable(tf.ones(1000)))

      # analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
      inference = ed.Gibbs({mu: qmu}, data={x: x_data})
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1e-2)
      self.assertAllClose(qmu.std().eval(), np.sqrt(1 / 51),
                          rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
  ed.set_seed(127832)
  tf.test.main()
