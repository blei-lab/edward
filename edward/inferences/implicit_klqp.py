from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.gan_inference import GANInference
from edward.models import RandomVariable
from edward.util import check_latent_vars, copy, get_session


class ImplicitKLqp(GANInference):
  """Variational inference with implicit probabilistic models
  [@tran2017deep].

  It minimizes the KL divergence

  $\\text{KL}( q(z, \\beta; \lambda) \| p(z, \\beta \mid x) ),$

  where $z$ are local variables associated to a data point and
  $\\beta$ are global variables shared across data points.

  Global latent variables require `log_prob()` and need to return a
  random sample when fetched from the graph. Local latent variables
  and observed variables require only a random sample when fetched
  from the graph. (This is true for both $p$ and $q$.)

  All variational factors must be reparameterizable: each of the
  random variables (`rv`) satisfies `rv.is_reparameterized` and
  `rv.is_continuous`.

  #### Notes

  Unlike `GANInference`, `discriminator` takes dict's as input,
  and must subset to the appropriate values through lexical scoping
  from the previously defined model and latent variables. This is
  necessary as the discriminator can take an arbitrary set of data,
  latent, and global variables.

  Note the type for `discriminator`'s output changes when one
  passes in the `scale` argument to `initialize()`.

  + If `scale` has at most one item, then `discriminator`
  outputs a tensor whose multiplication with that element is
  broadcastable. (For example, the output is a tensor and the single
  scale factor is a scalar.)
  + If `scale` has more than one item, then in order to scale
  its corresponding output, `discriminator` must output a
  dictionary of same size and keys as `scale`.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  def __init__(self, latent_vars, data=None, discriminator=None,
               global_vars=None):
    """Create an inference algorithm.

    Args:
      discriminator: function.
        Function (with parameters). Unlike `GANInference`, it is
        interpreted as a ratio estimator rather than a discriminator.
        It takes three arguments: a data dict, local latent variable
        dict, and global latent variable dict. As with GAN
        discriminators, it can take a batch of data points and local
        variables, of size $M$, and output a vector of length
        $M$.
      global_vars: dict of RandomVariable to RandomVariable.
        Identifying which variables in `latent_vars` are global
        variables, shared across data points. These will not be
        encompassed in the ratio estimation problem, and will be
        estimated with tractable variational approximations.
    """
    if not callable(discriminator):
      raise TypeError("discriminator must be a callable function.")

    self.discriminator = discriminator
    if global_vars is None:
      global_vars = {}

    check_latent_vars(global_vars)
    self.global_vars = global_vars
    # call grandparent's method; avoid parent (GANInference)
    super(GANInference, self).__init__(latent_vars, data)

  def initialize(self, ratio_loss='log', *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      ratio_loss: str or fn.
        Loss function minimized to get the ratio estimator. 'log' or 'hinge'.
        Alternatively, one can pass in a function of two inputs,
        `psamples` and `qsamples`, and output a point-wise value
        with shape matching the shapes of the two inputs.
    """
    if callable(ratio_loss):
      self.ratio_loss = ratio_loss
    elif ratio_loss == 'log':
      self.ratio_loss = log_loss
    elif ratio_loss == 'hinge':
      self.ratio_loss = hinge_loss
    else:
      raise ValueError('Ratio loss not found:', ratio_loss)

    return super(ImplicitKLqp, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Build loss function

    $-\Big(\mathbb{E}_{q(\\beta)} [\log p(\\beta) - \log q(\\beta) ] +
        \sum_{n=1}^N \mathbb{E}_{q(\\beta)q(z_n\mid\\beta)} [
            r^*(x_n, z_n, \\beta) ] \Big).$

    We minimize it with respect to parameterized variational
    families $q(z, \\beta; \lambda)$.

    $r^*(x_n, z_n, \\beta)$ is a function of a single data point
    $x_n$, single local variable $z_n$, and all global
    variables $\\beta$. It is equal to the log-ratio

    $\log p(x_n, z_n\mid \\beta) - \log q(x_n, z_n\mid \\beta),$

    where $q(x_n)$ is the empirical data distribution. Rather
    than explicit calculation, $r^*(x, z, \\beta)$ is the
    solution to a ratio estimation problem, minimizing the specified
    `ratio_loss`.

    Gradients are taken using the reparameterization trick
    [@kingma2014auto].

    #### Notes

    This also includes model parameters $p(x, z, \\beta; \\theta)$
    and variational distributions with inference networks
    $q(z\mid x)$.

    There are a bunch of extensions we could easily do in this
    implementation:

    + further factorizations can be used to better leverage the
      graph structure for more complicated models;
    + score function gradients for global variables;
    + use more samples; this would require the `copy()` utility
      function for q's as well, and an additional loop. we opt not to
      because it complicates the code;
    + analytic KL/swapping out the penalty term for the globals.
    """
    # Collect tensors used in calculation of losses.
    scope = tf.get_default_graph().unique_name("inference")
    qbeta_sample = {}
    pbeta_log_prob = 0.0
    qbeta_log_prob = 0.0
    for beta, qbeta in six.iteritems(self.global_vars):
      # Draw a sample beta' ~ q(beta) and calculate
      # log p(beta') and log q(beta').
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

    # Collect x' ~ p(x | z', beta') and x' ~ q(x).
    dict_swap = qbeta_sample.copy()
    dict_swap.update(qz_sample)
    x_psample = {}
    x_qsample = {}
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
        x_psample[x] = x_copy.value()
        x_qsample[x] = x_data

    with tf.variable_scope("Disc"):
      r_psample = self.discriminator(x_psample, pz_sample, qbeta_sample)

    with tf.variable_scope("Disc", reuse=True):
      r_qsample = self.discriminator(x_qsample, qz_sample, qbeta_sample)

    # Form ratio loss and ratio estimator.
    if len(self.scale) <= 1:
      loss_d = tf.reduce_mean(self.ratio_loss(r_psample, r_qsample))
      scale = list(six.itervalues(self.scale))
      scale = scale[0] if scale else 1.0
      scaled_ratio = tf.reduce_sum(scale * r_qsample)
    else:
      loss_d = [tf.reduce_mean(self.ratio_loss(r_psample[key], r_qsample[key]))
                for key in six.iterkeys(self.scale)]
      loss_d = tf.reduce_sum(loss_d)
      scaled_ratio = [tf.reduce_sum(self.scale[key] * r_qsample[key])
                      for key in six.iterkeys(self.scale)]
      scaled_ratio = tf.reduce_sum(scaled_ratio)

    reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
    reg_terms_all = tf.losses.get_regularization_losses()
    reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

    # Form variational objective.
    loss = -(pbeta_log_prob - qbeta_log_prob + scaled_ratio -
             tf.reduce_sum(reg_terms))
    loss_d = loss_d + tf.reduce_sum(reg_terms_d)

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
