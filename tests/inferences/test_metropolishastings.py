from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Empirical


class test_metropolishastings_class(tf.test.TestCase):

  def test_normalnormal_float32(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      n_samples = 2000
      qmu = Empirical(params=tf.Variable(tf.ones(n_samples)))

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.MetropolisHastings({mu: qmu},
                                        {mu: mu},
                                        data={x: x_data})
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-1, atol=1e-1)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-1, atol=1e-1)

      old_t, old_n_accept = sess.run([inference.t, inference.n_accept])
      self.assertEqual(old_t, n_samples)
      self.assertGreater(old_n_accept, 0.1)
      sess.run(inference.reset)
      new_t, new_n_accept = sess.run([inference.t, inference.n_accept])
      self.assertEqual(new_t, 0)
      self.assertEqual(new_n_accept, 0)

  def test_normalnormal_float32(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=tf.constant(0.0, dtype=tf.float64),
                  scale=tf.constant(1.0, dtype=tf.float64))
      x = Normal(loc=mu,
                 scale=tf.constant(1.0, dtype=tf.float64),
                 sample_shape=50)

      n_samples = 2000
      qmu = Empirical(params=tf.Variable(tf.ones(n_samples, dtype=tf.float64)))

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.MetropolisHastings({mu: qmu},
                                        {mu: mu},
                                        data={x: x_data})
      inference.run()

      self.assertAllClose(qmu.mean().eval(), 0, rtol=1e-1, atol=1e-1)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=1e-1, atol=1e-1)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
