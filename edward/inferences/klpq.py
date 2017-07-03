from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy


class KLpq(VariationalInference):
  """Variational inference with the KL divergence

  $\\text{KL}( p(z \mid x) \| q(z) ).$

  To perform the optimization, this class uses a technique from
  adaptive importance sampling (Cappe et al., 2008).

  #### Notes

  `KLpq` also optimizes any model parameters $p(z\mid x;
  \\theta)$. It does this by variational EM, minimizing

  $\mathbb{E}_{p(z \mid x; \lambda)} [ \log p(x, z; \\theta) ]$

  with respect to $\\theta$.

  In conditional inference, we infer $z` in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$. During gradient calculation, instead
  of using the model's density

  $\log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),$

  for each sample $s=1,\ldots,S$, `KLpq` uses

  $\log p(x, z^{(s)}, \\beta^{(s)}),$

  where $z^{(s)} \sim q(z; \lambda)$ and$\\beta^{(s)}
  \sim q(\\beta)$.
  """
  def __init__(self, *args, **kwargs):
    super(KLpq, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      n_samples: int, optional.
        Number of samples from variational model for calculating
        stochastic gradients.
    """
    self.n_samples = n_samples
    return super(KLpq, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Build loss function

    $\\text{KL}( p(z \mid x) \| q(z) )
      = \mathbb{E}_{p(z \mid x)} [ \log p(z \mid x) - \log q(z; \lambda) ]$

    and stochastic gradients based on importance sampling.

    The loss function can be estimated as

    $\\frac{1}{S} \sum_{s=1}^S [
      w_{\\text{norm}}(z^s; \lambda) (\log p(x, z^s) - \log q(z^s; \lambda) ],$

    where for $z^s \sim q(z; \lambda)$,

    $w_{\\text{norm}}(z^s; \lambda) =
          w(z^s; \lambda) / \sum_{s=1}^S w(z^s; \lambda)$

    normalizes the importance weights, $w(z^s; \lambda) = p(x,
    z^s) / q(z^s; \lambda)$.

    This provides a gradient,

    $- \\frac{1}{S} \sum_{s=1}^S [
      w_{\\text{norm}}(z^s; \lambda) \\nabla_{\lambda} \log q(z^s; \lambda) ].$
    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    for s in range(self.n_samples):
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      scope = base_scope + tf.get_default_graph().unique_name("sample")
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
            qz_copy.log_prob(tf.stop_gradient(dict_swap[z])))

      for z in six.iterkeys(self.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(dict_swap[x]))

    p_log_prob = tf.stack(p_log_prob)
    q_log_prob = tf.stack(q_log_prob)

    if self.logging:
      tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                        collections=[self._summary_key])
      tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                        collections=[self._summary_key])

    log_w = p_log_prob - q_log_prob
    log_w_norm = log_w - tf.reduce_logsumexp(log_w)
    w_norm = tf.exp(log_w_norm)

    loss = tf.reduce_mean(w_norm * log_w)
    grads = tf.gradients(
        -tf.reduce_mean(q_log_prob * tf.stop_gradient(w_norm)),
        var_list)
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars
