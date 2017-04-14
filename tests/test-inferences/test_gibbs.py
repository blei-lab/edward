from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical


class test_gibbs_class(tf.test.TestCase):

  def test_beta_bernoulli(self):
    with self.test_session() as sess:
      # DATA
      x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

      # MODEL
      p = Beta(a=1.0, b=1.0)
      x = Bernoulli(p=p, sample_shape=10)

      # INFERENCE
      qp = Empirical(tf.Variable(tf.zeros(1000)))
      inference = ed.Gibbs({p: qp}, data={x: x_data})
      inference.run()

      # CRITICISM
      true_posterior = Beta(a=3.0, b=9.0)

      val_est, val_true = sess.run([qp.mean(), true_posterior.mean()])
      self.assertAllClose(val_est, val_true, rtol=1e-2, atol=1e-2)

      val_est, val_true = sess.run([qp.variance(), true_posterior.variance()])
      self.assertAllClose(val_est, val_true, rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
  ed.set_seed(127832)
  tf.test.main()
