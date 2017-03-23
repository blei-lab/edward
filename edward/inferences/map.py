from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable, PointMass
from edward.util import copy


class MAP(VariationalInference):
  """Maximum a posteriori.

  This class implements gradient-based optimization to solve the
  optimization problem,

  .. math::

    \min_{z} - p(z \mid x).

  This is equivalent to using a ``PointMass`` variational distribution
  and minimizing the unnormalized objective,

  .. math::

    - \mathbb{E}_{q(z; \lambda)} [ \log p(x, z) ].

  Notes
  -----
  This class is currently restricted to optimization over
  differentiable latent variables. For example, it does not solve
  discrete optimization.

  This class also minimizes the loss with respect to any model
  parameters :math:`p(z \mid x; \\theta)`.

  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`. ``MAP`` optimizes
  :math:`\mathbb{E}_{q(\\beta)} [ \log p(x, z, \\beta) ]`, leveraging
  a single Monte Carlo sample, :math:`\log p(x, z, \\beta^*)`, where
  :math:`\\beta^* \sim q(\\beta)`. This is a lower bound to the
  marginal density :math:`\log p(x, z)`, and it is exact if
  :math:`q(\\beta) = p(\\beta \mid x)` (up to stochasticity).
  """
  def __init__(self, latent_vars=None, data=None):
    """
    Parameters
    ----------
    latent_vars : list of RandomVariable or
                  dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized
      using a ``PointMass`` random variable that is defined
      internally (with unconstrained support). If dictionary, each
      value in the dictionary must be a ``PointMass`` random variable.

    Examples
    --------
    Most explicitly, ``MAP`` is specified via a dictionary:

    >>> qpi = PointMass(params=ed.to_simplex(tf.Variable(tf.zeros(K-1))))
    >>> qmu = PointMass(params=tf.Variable(tf.zeros(K*D)))
    >>> qsigma = PointMass(params=tf.nn.softplus(tf.Variable(tf.zeros(K*D))))
    >>> ed.MAP({pi: qpi, mu: qmu, sigma: qsigma}, data)

    We also automate the specification of ``PointMass`` distributions,
    so one can pass in a list of latent variables instead:

    >>> ed.MAP([beta], data)
    >>> ed.MAP([pi, mu, sigma], data)

    Currently, ``MAP`` can only instantiate ``PointMass`` random variables
    with unconstrained support. To constrain their support, one must
    manually pass in the ``PointMass`` family.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("posterior"):
        latent_vars = {rv: PointMass(
            params=tf.Variable(tf.random_normal(rv.get_batch_shape())))
            for rv in latent_vars}
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(qz, PointMass):
          raise TypeError("Posterior approximation must consist of only "
                          "PointMass random variables.")

    super(MAP, self).__init__(latent_vars, data)

  def build_loss_and_gradients(self, var_list):
    """Build loss function. Its automatic differentiation
    is the gradient of

    .. math::
      - \log p(x,z)
    """
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = 'inference_' + str(id(self))
    dict_swap = {z: qz.value()
                 for z, qz in six.iteritems(self.latent_vars)}
    for x, qx in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          dict_swap[x] = qx.value()
        else:
          dict_swap[x] = qx

    p_log_prob = 0.0
    for z in six.iterkeys(self.latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_prob += tf.reduce_sum(
          self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(self.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_prob += tf.reduce_sum(
            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

    loss = -p_log_prob

    grads = tf.gradients(loss, [v._ref() for v in var_list])
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars
