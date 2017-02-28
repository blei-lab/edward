from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.gan_inference import GANInference
from edward.models import RandomVariable
from edward.util import copy, get_session


# TODO what to call this?
class ImplicitKLqp(GANInference):
  """Variational inference with implicit probabilistic models.

  It minimizes the KL divergence

  .. math::

    \\text{KL}( q(z; \lambda) \| p(z \mid x) ).

  Global latent variables require ``log_prob()`` and need to return a
  random sample when fetched from the graph. Local latent variables
  and observed variables require only a random sample when fetched
  from the graph. (This is true for both the probability model and
  variational model.)
  """
  def __init__(self, latent_vars, data=None, discriminator=None,
               global_vars=None):
    """
    Parameters
    ----------
    discriminator : function
      Function (with parameters) to discriminate samples. It should
      output logit probabilities (real-valued) and not probabilities
      in [0, 1]. Unlike ``GANInference``, it takes three arguments: a
      data dict, local latent variable dict, and global latent
      variable dict. As with GAN discriminators, it can take a batch
      of data points and local variables, of size M, and output a
      vector of length M.
    global_vars : dict of RandomVariable to RandomVariable, optional
      Identifying which variables in ``latent_vars`` are global
      variables, shared across data points. These will not be
      encompassed in the ratio estimation problem, and will be
      estimated with tractable variational approximations.

    Notes
    -----
    Unlike ``GANInference`` D takes dict's as input, and must subset
    to the appropriate values through lexical scoping from the
    previously defined model and latent variables. This is necessary
    as the discriminator can take an arbitrary set of data, latent, and
    global variables.
    """
    if discriminator is None:
      raise NotImplementedError()

    if global_vars is None:
      global_vars = {}
    elif not isinstance(latent_vars, dict):
      raise TypeError()

    self.discriminator = discriminator
    self.global_vars = global_vars
    # call grandparent's method; avoid parent (GANInference)
    super(GANInference, self).__init__(latent_vars, data, model_wrapper=None)

  def build_loss_and_gradients(self, var_list):
    """Build loss function

    .. math::

      -[ \mathbb{E}_{q(\beta)} [
           \sum_{n=1}^N \mathbb{E}_{q(z_n | \beta)} [ -D*(x_n, z_n, \beta) ] ] +
         \mathbb{E}_{q(\beta)} [ log p(\beta) - log q(\beta) ] ].

    We minimize it with respect to parameterized variational
    families :math:`q(z, beta; \lambda)`.

    :math:`D*(x_n, z_n, beta)` is a function of a single data point
    :math:`x_n`, single local variable :math:`z_n`, and all global
    variables :math:`\beta`. It is equal to the log-ratio

    .. math::

      \log q(z_n | \beta) - \log p(x_n, z_n | \beta)

    Rather than explicit calculation, :math:`D*(x, z, \beta)` is the
    solution to an estimation problem

    .. math::

      \\text{argmin}_D
      \mathbb{E}_{p(x_n)q(z_n | \beta)} [ log D(x_n, z_n, \beta) ] +
      \mathbb{E}_{p(x_n, z_n | \beta)} [ log (1 - D(x_n, z_n, \beta)) ].

    Gradients are taken using the reparameterization trick (Kingma and
    Welling, 2014).

    This also includes model parameters :math:`p(x, z, beta; theta)`
    and variational distributions with inference networks :math:`q(z |
    x)`.

    There are a bunch of extensions we could easily do in this
    implementation:
    + further factorizations can be used to better leverage the
    graph structure for more complicated models;
    + use more samples; this would require the ``copy()`` utility
    function for q's as well, and an additional loop. we opt not to
    because it complicates the code.
    + analytic KL/swapping out the penalty term for the globals.
    + various extensions of hierarchical models, and not just an
    explicit local and global var distinction.
    """
    # Collect tensors used in calculation of losses.
    scope = 'inference_' + str(id(self))
    qbeta_sample = {}
    pbeta_log_prob = 0.0
    qbeta_log_prob = 0.0
    for beta, qbeta in six.iteritems(self.global_vars):
      # Draw a sample beta' ~ q(beta) and calculate log p(beta') and log q(beta').
      qbeta_sample[beta] = qbeta.value()
      pbeta_log_prob += tf.reduce_sum(beta.log_prob(qbeta_sample[beta]))
      qbeta_log_prob += tf.reduce_sum(qbeta.log_prob(qbeta_sample[beta]))

    pz_sample = {}
    qz_sample = {}
    for z, qz in six.iteritems(self.latent_vars):
      if z not in self.global_vars:
        # Copy local variables p(z), q(z) to draw samples
        # z' ~ p(z | beta'), z' ~ q(z | beta').
        pz_copy = copy(z, dict_swap=qbeta_sample, scope=scope)
        pz_sample[z] = pz_copy.value()
        qz_sample[z] = qz.value()

    # Collect x' ~ p(x | z', beta') and x' ~ p^*(x).
    dict_swap = qbeta_sample.copy()
    dict_swap.update(qz_sample)
    x_psample = {}
    x_qsample = {}
    debug_with_true_ratio = True
    if debug_with_true_ratio:
      p_log_lik = 0.0
    for x, x_data in six.iteritems(self.data):
      if isinstance(x, tf.Tensor):
        if "Placeholder" not in x.op.type:
          # Copy p(x | z, beta) to get draw p(x | z', beta').
          x_copy = copy(x, dict_swap=dict_swap, scope=scope)
          x_psample[x] = x_copy
          x_qsample[x] = x_data
      elif isinstance(x, RandomVariable):
        # Copy p(x | z, beta) to get draw p(x | z', beta').
        x_copy = copy(x, dict_swap=dict_swap, scope=scope)
        if debug_with_true_ratio:
          p_log_lik = tf.reduce_sum(
              self.scale.get(x, 1.0) * x_copy.log_prob(x_data))
        x_psample[x] = x_copy.value()
        x_qsample[x] = x_data

    with tf.variable_scope("Disc"):
      r_psample = self.discriminator(x_psample, pz_sample, qbeta_sample)

    with tf.variable_scope("Disc", reuse=True):
      r_qsample = self.discriminator(x_qsample, qz_sample, qbeta_sample)

    # Form variational objective and auxiliary log-ratio loss.
    # TODO hard to know how to scale D if there is more than one
    # thing being scaled; e.g, since it's scaling the f(x, z), it
    # should be [scale[x], scale[z]] * f(x, z), assuming x, z have
    # the same shape to begin with.
    # + D should generally output a dict, of same size as its input
    #   + this is impractical though
    # + for now, scale all values with just the first scale
    # argument, and hope it's a tensor broadcastable to all output
    # (such as a scalar)
    scale = list(six.itervalues(self.scale))
    scale = scale[0] if scale else 1.0
    loss = -(tf.reduce_sum(scale * r_qsample) + pbeta_log_prob - qbeta_log_prob)
    if debug_with_true_ratio:
      self.ratio_true = scale * p_log_lik
      self.ratio_est = tf.reduce_sum(scale * r_qsample)

    loss_d = log_loss(r_psample, r_qsample)
    loss_d = tf.reduce_mean(loss_d)

    var_list_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    if var_list is None:
      var_list = [v for v in tf.trainable_variables() if v not in var_list_d]

    grads = tf.gradients(loss, var_list)
    grads_d = tf.gradients(loss_d, var_list_d)
    grads_and_vars = list(zip(grads, var_list))
    grads_and_vars_d = list(zip(grads_d, var_list_d))
    return loss, grads_and_vars, loss_d, grads_and_vars_d


def log_loss(psample, qsample):
  """Point-wise log loss."""
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(psample), logits=psample) + \
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(qsample), logits=qsample)
  return loss

def hinge_loss(psample, qsample):
  """Point-wise hinge loss."""
  loss = tf.nn.relu(1.0 - psample) + tf.nn.relu(1.0 + qsample)
  return loss
