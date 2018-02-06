from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal


class test_klqp_class(tf.test.TestCase):

  def _test_normal_normal(self, Inference, default, *args, **kwargs):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      if not default:
        qmu_loc = tf.Variable(tf.random_normal([]))
        qmu_scale = tf.nn.softplus(tf.Variable(tf.random_normal([])))
        qmu = Normal(loc=qmu_loc, scale=qmu_scale)

        # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
        inference = Inference({mu: qmu}, data={x: x_data})
      else:
        inference = Inference([mu], data={x: x_data})
        qmu = inference.latent_vars[mu]
      inference.run(*args, **kwargs)

      self.assertAllClose(qmu.mean().eval(), 0, rtol=0.1, atol=0.6)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=0.15, atol=0.5)

      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer')
      old_t, old_variables = sess.run([inference.t, variables])
      self.assertEqual(old_t, inference.n_iter)
      sess.run(inference.reset)
      new_t, new_variables = sess.run([inference.t, variables])
      self.assertEqual(new_t, 0)
      self.assertNotEqual(old_variables, new_variables)

  def _test_model_parameter(self, Inference, *args, **kwargs):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

      p = tf.sigmoid(tf.Variable(0.5))
      x = Bernoulli(probs=p, sample_shape=10)

      inference = Inference({}, data={x: x_data})
      inference.run(*args, **kwargs)

      self.assertAllClose(p.eval(), 0.2, rtol=5e-2, atol=5e-2)

  def test_klqp(self):
    self._test_normal_normal(ed.KLqp, default=False, n_iter=5000)
    self._test_normal_normal(ed.KLqp, default=True, n_iter=5000)
    self._test_model_parameter(ed.KLqp, n_iter=50)

  def test_klqp_nsamples_check(self):
    with self.assertRaisesRegexp(ValueError,
                                 "n_samples should be greater than zero: 0"):
      self._test_normal_normal(ed.KLqp, default=True, n_samples=0, n_iter=10)
      self._test_normal_normal(ed.ReparameterizationEntropyKLqp,
                               default=True, n_samples=0, n_iter=10)
      self._test_normal_normal(ed.ReparameterizationKLqp,
                               default=True, n_samples=0, n_iter=10)
      self._test_normal_normal(ed.ReparameterizationKLKLqp,
                               default=True, n_samples=0, n_iter=10)
      self._test_normal_normal(ed.ScoreEntropyKLqp,
                               default=True, n_samples=0, n_iter=10)
      self._test_normal_normal(ed.ScoreKLqp,
                               default=True, n_samples=0, n_iter=10)
      self._test_normal_normal(ed.ScoreKLKLqp,
                               default=True, n_samples=0, n_iter=10)
      self._test_normal_normal(ed.ScoreRBKLqp,
                               default=True, n_samples=0, n_iter=10)

  def test_reparameterization_entropy_klqp(self):
    self._test_normal_normal(
        ed.ReparameterizationEntropyKLqp, default=False, n_iter=5000)
    self._test_normal_normal(
        ed.ReparameterizationEntropyKLqp, default=True, n_iter=5000)
    self._test_model_parameter(ed.ReparameterizationEntropyKLqp, n_iter=50)

  def test_reparameterization_klqp(self):
    self._test_normal_normal(
        ed.ReparameterizationKLqp, default=False, n_iter=5000)
    self._test_normal_normal(
        ed.ReparameterizationKLqp, default=True, n_iter=5000)
    self._test_model_parameter(ed.ReparameterizationKLqp, n_iter=50)

  def test_reparameterization_kl_klqp(self):
    self._test_normal_normal(
        ed.ReparameterizationKLKLqp, default=False, n_iter=5000)
    self._test_normal_normal(
        ed.ReparameterizationKLKLqp, default=True, n_iter=5000)
    self._test_model_parameter(ed.ReparameterizationKLKLqp, n_iter=50)

  def test_score_entropy_klqp(self):
    self._test_normal_normal(
        ed.ScoreEntropyKLqp, default=False, n_samples=5, n_iter=5000)
    self._test_normal_normal(
        ed.ScoreEntropyKLqp, default=True, n_samples=5, n_iter=5000)
    self._test_model_parameter(ed.ScoreEntropyKLqp, n_iter=50)

  def test_score_klqp(self):
    self._test_normal_normal(
        ed.ScoreKLqp, default=False, n_samples=5, n_iter=5000)
    self._test_normal_normal(
        ed.ScoreKLqp, default=True, n_samples=5, n_iter=5000)
    self._test_model_parameter(ed.ScoreKLqp, n_iter=50)

  def test_score_kl_klqp(self):
    self._test_normal_normal(
        ed.ScoreKLKLqp, default=False, n_samples=5, n_iter=5000)
    self._test_normal_normal(
        ed.ScoreKLKLqp, default=True, n_samples=5, n_iter=5000)
    self._test_model_parameter(ed.ScoreKLKLqp, n_iter=50)

  def test_score_rb_klqp(self):
    self._test_normal_normal(
        ed.ScoreRBKLqp, default=False, n_samples=5, n_iter=5000)
    self._test_normal_normal(
        ed.ScoreRBKLqp, default=True, n_samples=5, n_iter=5000)
    self._test_model_parameter(ed.ScoreRBKLqp, n_iter=50)

if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
