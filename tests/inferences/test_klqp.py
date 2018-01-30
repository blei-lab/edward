from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, Dirichlet, Multinomial, \
  Gamma, Poisson

from edward.inferences.klqp import build_rejection_sampling_loss_and_gradients


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

      self.assertAllClose(qmu.mean().eval(), 0, rtol=0.15, atol=0.5)
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

  def _test_poisson_gamma(self, Inference, *args, **kwargs):
      with self.test_session() as sess:
        x_data = np.array([2, 8, 3, 6, 1], dtype=np.int32)

        rate = Gamma(5.0, 1.0)
        x = Poisson(rate=rate, sample_shape=5)

        qalpha = tf.nn.softplus(tf.Variable(tf.random_normal([]), name='qalpha'))
        qbeta = tf.nn.softplus(tf.Variable(tf.random_normal([]), name='qbeta'))
        qgamma = Gamma(qalpha, qbeta, allow_nan_stats=False)

        # sum(x_data) = 20
        # len(x_data) = 5
        # analytic solution: Gamma(alpha=5+20, beta=1+5)
        inference = Inference({rate: qgamma}, data={x: x_data})

        inference.run(*args, **kwargs)

        self.assertAllClose(tf.nn.softplus(qalpha).eval(), 25., atol=1e-2)
        self.assertAllClose(tf.nn.softplus(qbeta).eval(), 6., atol=1e-2)

  def _test_multinomial_dirichlet(self, Inference, *args, **kwargs):
      with self.test_session() as sess:
        x_data = tf.constant([2, 7, 1], dtype=np.float32)

        probs = Dirichlet([1., 1., 1.])
        x = Multinomial(total_count=10.0, probs=probs)

        qalpha = tf.Variable(tf.random_normal([3]))
        qprobs = Dirichlet(qalpha)

        # analytic solution: Dirichlet(alpha=[1+2, 1+7, 1+1])
        inference = Inference({probs: qprobs}, data={x: x_data})

        inference.run(*args, **kwargs)

  def _test_build_rejection_sampling_loss_and_gradients(self, *args, **kwargs):
    with self.test_session() as sess:
      x_data = np.array([3, 3, 3, 3, 0], dtype=np.float32)

      rate = Gamma(1.0, 1.0)
      x = Poisson(rate=rate, sample_shape=5)

      _qalpha = tf.Variable(-0.52817175, name='qalpha')
      _qbeta = tf.Variable(-1.07296862, name='qbeta')
      var_list = [_qalpha, _qbeta]

      qalpha = tf.exp(_qalpha) + 1
      qbeta = tf.exp(_qbeta)
      qgamma = Gamma(qalpha, qbeta, allow_nan_stats=False)

      class DummyInference:
        n_samples = 1
        data = {x: x_data}
        latent_vars = {rate: qgamma}
        scale = {}
        logging = False

      tf.global_variables_initializer().run()

      expected_g_reparam = np.array([-10.348131898560453, 31.81539831675293])
      expected_g_score = np.array([0.30550423741109256, 0.0])
      expected_g_entropy = np.array([0.28863888798339055, -1.0])

      loss, grads_and_vars = build_rejection_sampling_loss_and_gradients(DummyInference(),
        var_list, epsilon=tf.constant(0.86540763))

      self.assertAllClose([g.eval() for g, v in grads_and_vars],
        expected_g_reparam + expected_g_score + expected_g_entropy, rtol=1e-6, atol=1e-6)

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

  def test_rejection_sampling_klqp(self):
    self._test_build_rejection_sampling_loss_and_gradients()
    self._test_poisson_gamma(
      ed.RejectionSamplingKLqp,
      n_samples=1,
      n_iter=50,
      optimizer='rmsprop',
      global_step=tf.Variable(0, trainable=False, name="global_step")
    )
    # self._test_multinomial_dirichlet(
    #   ed.RejectionSamplingKLqp, n_samples=5, n_iter=5000)


if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
