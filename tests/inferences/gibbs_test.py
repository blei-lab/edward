from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical, Normal


class test_gibbs_class(tf.test.TestCase):

  def _test_normal_normal(self, default, dtype):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=tf.constant(0.0, dtype=dtype),
                  scale=tf.constant(1.0, dtype=dtype))
      x = Normal(loc=mu, scale=tf.constant(1.0, dtype=dtype),
                 sample_shape=50)

      n_samples = 2000
      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      if not default:
        qmu = Empirical(params=tf.Variable(tf.ones(n_samples, dtype=dtype)))
        inference = ed.Gibbs({mu: qmu}, data={x: x_data})
      else:
        inference = ed.Gibbs([mu], data={x: x_data})
        qmu = inference.latent_vars[mu]
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-1, atol=1e-1)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-1, atol=1e-1)

      old_t, old_n_accept = sess.run([inference.t, inference.n_accept])
      if not default:
        self.assertEqual(old_t, n_samples)
      else:
        self.assertEqual(old_t, 1e4)
      self.assertGreater(old_n_accept, 0.1)
      sess.run(inference.reset)
      new_t, new_n_accept = sess.run([inference.t, inference.n_accept])
      self.assertEqual(new_t, 0)
      self.assertEqual(new_n_accept, 0)

  def test_beta_bernoulli(self):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

      p = Beta(1.0, 1.0)
      x = Bernoulli(probs=p, sample_shape=10)

      qp = Empirical(tf.Variable(tf.zeros(1000)))
      inference = ed.Gibbs({p: qp}, data={x: x_data})
      inference.run()

      true_posterior = Beta(3.0, 9.0)

      val_est, val_true = sess.run([qp.mean(), true_posterior.mean()])
      self.assertAllClose(val_est, val_true, rtol=1e-2, atol=1e-2)

      val_est, val_true = sess.run([qp.variance(), true_posterior.variance()])
      self.assertAllClose(val_est, val_true, rtol=1e-2, atol=1e-2)

  def test_normal_normal(self):
    self._test_normal_normal(True, tf.float32)
    self._test_normal_normal(False, tf.float32)
    self._test_normal_normal(True, tf.float64)
    self._test_normal_normal(False, tf.float64)

  def test_data_tensor(self):
    with self.test_session() as sess:
      x_data = tf.zeros(50)

      mu = Normal(0.0, 1.0)
      x = Normal(mu, 1.0, sample_shape=50)

      qmu = Empirical(tf.Variable(tf.ones(1000)))

      # analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
      inference = ed.Gibbs({mu: qmu}, data={x: x_data})
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-2, atol=1e-2)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
  ed.set_seed(127832)
  tf.test.main()
