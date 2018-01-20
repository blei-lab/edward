from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, Dirichlet, Multinomial, \
  Gamma, Poisson


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

          qalpha = tf.nn.softplus(tf.Variable(tf.random_normal([])))
          qbeta = tf.nn.softplus(tf.Variable(tf.random_normal([])))
          qgamma = Gamma(qalpha, qbeta)

          # Gamma rejection sampler variables
          def gamma_reparam_func(epsilon, alpha, beta):

            def _gamma_reparam_func(alpha=alpha, beta=beta, epsilon=epsilon):
              a = alpha - (1. / 3)
              b = tf.sqrt(9 * alpha - 3)
              c = 1 + (epsilon / b)
              z = a * c**3
              return z

            def _gamma_reparam_func_alpha_lt_1(alpha=alpha, beta=beta, epsilon=epsilon):
              z_tilde = _gamma_reparam_func(alpha=alpha + 1, beta=beta)
              u = np.random.uniform()
              z = u ** (1 / alpha) * z_tilde
              return z

            z = tf.cond(tf.less(alpha, 1.), _gamma_reparam_func_alpha_lt_1, _gamma_reparam_func)
            z = tf.cond(tf.equal(beta, 1.), lambda: z, lambda: tf.divide(z, beta))
            return z

          gamma_rejection_sampler_vars = {
            'reparam_func': gamma_reparam_func,
            'epsilon_likelihood': Normal(loc=0.0, scale=1.0),
            'm': 10.
          }

          # sum(x_data) = 20
          # len(x_data) = 5
          # analytic solution: Gamma(alpha=5+20, beta=1+5)
          inference = Inference({rate: qgamma}, data={x: x_data},
              rejection_sampler_vars={Gamma: gamma_rejection_sampler_vars})

          inference.run(*args, **kwargs)

          import pdb; pdb.set_trace()

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
    self._test_poisson_gamma(
      ed.RejectionSamplingKLqp, n_samples=1, n_iter=5000)
    # self._test_multinomial_dirichlet(
    #   ed.RejectionSamplingKLqp, n_samples=5, n_iter=5000)

  def test_kucukelbir_grad(self):
    expected_grads_and_vars = [
      [(3.1018744, 1.0), (1.5509372, 2.0)],
      [(2.7902498, 0.84341073), (1.241244, 1.8959416)],
      [(2.6070995, 0.694741731383), (1.0711095, 1.80355358733)]
    ]
    t = 0.1
    delta = 10e-3
    eta = 1e-1

    def alp_optimizer_apply_gradients(n, s_n, grads_and_vars):
      ops = []
      for i, (grad, var) in enumerate(grads_and_vars):
        updated_s_n = s_n[i].assign( (t * grad**2) + (1 - t) * s_n[i] )

        p_n_first = eta * n**(-.5 + delta)
        p_n_second = (1 + tf.sqrt(updated_s_n[i]))**(-1)
        p_n = p_n_first * p_n_second

        updated_var = var.assign_add(-p_n * grad)
        ops.append((updated_s_n[i], p_n_first, p_n_second, p_n, updated_var))
      return ops

    with self.test_session() as sess:
      w1 = tf.Variable(tf.constant(1.))
      w2 = tf.Variable(tf.constant(2.))
      var_list = [w1, w2]

      x = tf.constant([3., 4., 5.])
      y = tf.constant([.8, .1, .1])

      pred = tf.nn.softmax(x * w1 * w2)
      loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
      grads = tf.gradients(loss, var_list)
      grads_and_vars = list(zip(grads, var_list))

      s_n = tf.Variable(tf.zeros(2))
      n = tf.Variable(tf.constant(1.))

      train = alp_optimizer_apply_gradients(n, s_n, grads_and_vars)
      increment_n = n.assign_add(1.)

      init = tf.global_variables_initializer()
      sess.run(init)

      for i in range(3):
        actual_grads_and_vars = sess.run(grads_and_vars)
        self.assertAllClose(
          actual_grads_and_vars, expected_grads_and_vars[i], rtol=5e-2, atol=5e-2)
        _ = sess.run(train)
        _ = sess.run(increment_n)


if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
