from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class RenyiDivergence(VariationalInference):
  """Variational inference with the Renyi divergence [@li2016renyi].

  It minimizes the Renyi divergence

  $ \text{D}_{R}^{(\alpha)}(q(z)||p(z \mid x))
      = \frac{1}{\alpha-1} \log \int q(z)^{\alpha} p(z \mid x)^{1-\alpha} dz.$

  The optimization is performed using the gradient estimator as defined in
  @li2016renyi.

  #### Notes
  + The gradient estimator used here does not have any analytic version.
  + The gradient estimator used here does not have any version for non
  reparametrizable models.
  + backward_pass = 'max': (extreme case $\alpha \rightarrow -\infty$)
  the algorithm chooses the sample that has the maximum unnormalised
  importance weight. This does not minimize the Renyi divergence
  anymore.
  + backward_pass = 'min': (extreme case $\alpha \rightarrow +\infty$)
  the algorithm chooses the sample that has the minimum unnormalised
  importance weight. This does not minimize the Renyi divergence
  anymore. This mode is not describe in the paper but implemented
  in the publicly available implementation of the paper's experiments.
  """

  def __init__(self, *args, **kwargs):

    super(RenyiDivergence, self).__init__(*args, **kwargs)

    is_reparameterizable = all([
        rv.reparameterization_type ==
        tf.contrib.distributions.FULLY_REPARAMETERIZED
        for rv in six.itervalues(self.latent_vars)])

    if not is_reparameterizable:
      raise NotImplementedError(
          "Variational Renyi inference only works with reparameterizable"
          " models.")

  def initialize(self,
                 n_samples=32,
                 alpha=1.0,
                 backward_pass='full',
                 *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
        n_samples: int, optional.
            Number of samples from variational model for calculating
            stochastic gradients.
        alpha: float, optional.
            Renyi divergence coefficient. $\alpha \in \mathbb{R}$.
            When $\alpha < 0$, the algorithm still does something sensible but
            does not minimize the Renyi divergence anymore.
            (see [@li2016renyi] - section 4.2)
        backward_pass: str, optional.
            Backward pass mode to be used.
            Options: 'min', 'max', 'full'
            (see [@li2016renyi] - section 4.2)
    """
    self.n_samples = n_samples
    self.alpha = alpha
    self.backward_pass = backward_pass

    return super(RenyiDivergence, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Build the Renyi ELBO function.

    Its automatic differentiation is a stochastic gradient of

    $ \mcalL_{R}^{\alpha}(q; x) =
            \frac{1}{1-\alpha} \log \dsE_{q} \left[
                \left( \frac{p(x, z)}{q(z)}\right)^{1-\alpha} \right].$

    It uses:

    + Monte Carlo approximation of the ELBO [@li2016renyi].
    + Reparameterization gradients [@kingma2014auto].
    + Stochastic approximation of the joint distribution [@li2016renyi].

    #### Notes

    + If the model is not reparameterizable, it returns a
    NotImplementedError.
    + See Renyi Divergence Variational Inference [@li2016renyi] for
    more details.
    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    for s in range(self.n_samples):
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      scope = base_scope \
          + tf.get_default_graph().unique_name("sample")
      dict_swap = {}
      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope=scope)
        dict_swap[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))

      for z in six.iterkeys(self.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(
              self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

    log_ratios = [p - q for p, q in zip(p_log_prob, q_log_prob)]

    if self.backward_pass == 'max':
      loss = tf.reduce_max(log_ratios, 0)
    elif self.backward_pass == 'min':
      loss = tf.reduce_min(log_ratios, 0)
    elif np.abs(self.alpha - 1.0) < 10e-3:
      loss = tf.reduce_mean(log_ratios)
    else:
      log_ratios = tf.stack(log_ratios)
      log_ratios = log_ratios * (1 - self.alpha)
      log_ratios_max = tf.reduce_max(log_ratios, 0)
      log_ratios = tf.log(
          tf.maximum(1e-9,
                     tf.reduce_mean(tf.exp(log_ratios - log_ratios_max), 0)))
      log_ratios = (log_ratios + log_ratios_max) / (1 - self.alpha)
      loss = tf.reduce_mean(log_ratios)
    loss = -loss

    if self.logging:
      p_log_prob = tf.reduce_mean(p_log_prob)
      q_log_prob = tf.reduce_mean(q_log_prob)
      tf.summary.scalar("loss/p_log_prob", p_log_prob,
                        collections=[self._summary_key])
      tf.summary.scalar("loss/q_log_prob", q_log_prob,
                        collections=[self._summary_key])

    grads = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars
