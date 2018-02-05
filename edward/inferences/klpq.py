from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy, get_descendants

try:
  from edward.models import Normal
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class KLpq(VariationalInference):
  """Variational inference with the KL divergence

  $\\text{KL}( p(z \mid x) \| q(z) ).$

  To perform the optimization, this class uses a technique from
  adaptive importance sampling [@oh1992adaptive].

  #### Notes

  `KLpq` also optimizes any model parameters $p(z\mid x;
  \\theta)$. It does this by variational EM, maximizing

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

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  def __init__(self, latent_vars=None, data=None):
    """Create an inference algorithm.

    Args:
      latent_vars: list of RandomVariable or
                   dict of RandomVariable to RandomVariable.
        Collection of random variables to perform inference on. If
        list, each random variable will be implictly optimized using a
        `Normal` random variable that is defined internally with a
        free parameter per location and scale and is initialized using
        standard normal draws. The random variables to approximate
        must be continuous.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope(None, default_name="posterior"):
        latent_vars_dict = {}
        continuous = \
            ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
        for z in latent_vars:
          if not hasattr(z, 'support') or z.support not in continuous:
            raise AttributeError(
                "Random variable {} is not continuous or a random "
                "variable with supported continuous support.".format(z))
          batch_event_shape = z.batch_shape.concatenate(z.event_shape)
          loc = tf.Variable(tf.random_normal(batch_event_shape))
          scale = tf.nn.softplus(
              tf.Variable(tf.random_normal(batch_event_shape)))
          latent_vars_dict[z] = Normal(loc=loc, scale=scale)
        latent_vars = latent_vars_dict
        del latent_vars_dict

    super(KLpq, self).__init__(latent_vars, data)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      n_samples: int.
        Number of samples from variational model for calculating
        stochastic gradients.
    """
    if n_samples <= 0:
      raise ValueError(
          "n_samples should be greater than zero: {}".format(n_samples))
    self.n_samples = n_samples
    return super(KLpq, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Build loss function

    $\\text{KL}( p(z \mid x) \| q(z) )
      = \mathbb{E}_{p(z \mid x)} [ \log p(z \mid x) - \log q(z; \lambda) ]$

    and stochastic gradients based on importance sampling.

    The loss function can be estimated as

    $\sum_{s=1}^S [
      w_{\\text{norm}}(z^s; \lambda) (\log p(x, z^s) - \log q(z^s; \lambda) ],$

    where for $z^s \sim q(z; \lambda)$,

    $w_{\\text{norm}}(z^s; \lambda) =
          w(z^s; \lambda) / \sum_{s=1}^S w(z^s; \lambda)$

    normalizes the importance weights, $w(z^s; \lambda) = p(x,
    z^s) / q(z^s; \lambda)$.

    This provides a gradient,

    $- \sum_{s=1}^S [
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
    reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

    if self.logging:
      tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                        collections=[self._summary_key])
      tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                        collections=[self._summary_key])
      tf.summary.scalar("loss/reg_penalty", reg_penalty,
                        collections=[self._summary_key])

    log_w = p_log_prob - q_log_prob
    log_w_norm = log_w - tf.reduce_logsumexp(log_w)
    w_norm = tf.exp(log_w_norm)
    loss = tf.reduce_sum(w_norm * log_w) - reg_penalty

    q_rvs = list(six.itervalues(self.latent_vars))
    q_vars = [v for v in var_list
              if len(get_descendants(tf.convert_to_tensor(v), q_rvs)) != 0]
    q_grads = tf.gradients(
        -(tf.reduce_sum(q_log_prob * tf.stop_gradient(w_norm)) - reg_penalty),
        q_vars)
    p_vars = [v for v in var_list if v not in q_vars]
    p_grads = tf.gradients(-loss, p_vars)
    grads_and_vars = list(zip(q_grads, q_vars)) + list(zip(p_grads, p_vars))
    return loss, grads_and_vars
