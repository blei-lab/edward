from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.map import MAP
from edward.models import \
    MultivariateNormalCholesky, MultivariateNormalDiag, \
    MultivariateNormalFull, PointMass, RandomVariable
from edward.util import get_session, get_variables


class Laplace(MAP):
  """Laplace approximation (Laplace, 1774).

  It approximates the posterior distribution using a multivariate
  normal distribution centered at the mode of the posterior.

  We implement this by running ``MAP`` to find the posterior mode.
  This forms the mean of the normal approximation. We then compute the
  inverse Hessian at the mode of the posterior. This forms the
  covariance of the normal approximation.
  """
  def __init__(self, latent_vars, data=None):
    """
    Parameters
    ----------
    latent_vars : list of RandomVariable or
                  dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on. If list,
      each random variable will be implictly optimized using a
      ``MultivariateNormalCholesky`` random variable that is defined
      internally (with unconstrained support). If dictionary, each
      random variable must be a ``MultivariateNormalCholesky``,
      ``MultivariateNormalFull``, or ``MultivariateNormalDiag`` random
      variable.

    Notes
    -----
    If ``MultivariateNormalDiag`` random variables are specified as
    approximations, then the Laplace approximation will only produce
    the diagonal. This does not capture correlation among the
    variables but it does not require a potentially expensive matrix
    inversion.

    Examples
    --------
    >>> X = tf.placeholder(tf.float32, [N, D])
    >>> w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
    >>> y = Normal(mu=ed.dot(X, w), sigma=tf.ones(N))
    >>>
    >>> qw = MultivariateNormalFull(mu=tf.Variable(tf.random_normal([D])),
    >>>                             sigma=tf.Variable(tf.random_normal([D, D])))
    >>>
    >>> inference = ed.Laplace({w: qw}, data={X: X_train, y: y_train})
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("posterior"):
        latent_vars = {rv: MultivariateNormalCholesky(
            mu=tf.Variable(tf.random_normal(rv.batch_shape())),
            chol=tf.Variable(tf.random_normal(
                rv.get_batch_shape().concatenate(rv.get_batch_shape()[-1]))))
            for rv in latent_vars}
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(
            qz, (MultivariateNormalCholesky, MultivariateNormalDiag,
                 MultivariateNormalFull)):
          raise TypeError("Posterior approximation must consist of only "
                          "MultivariateCholesky, MultivariateNormalDiag, "
                          "or MultivariateNormalFull random variables.")

    # call grandparent's method; avoid parent (MAP)
    super(MAP, self).__init__(latent_vars, data)

  def initialize(self, var_list=None, *args, **kwargs):
    # Store latent variables in a temporary attribute; MAP will
    # optimize ``PointMass`` random variables, which subsequently
    # optimizes mean parameters of the normal approximations.
    self.latent_vars_normal = self.latent_vars.copy()
    self.latent_vars = {z: PointMass(params=qz.mu)
                        for z, qz in six.iteritems(self.latent_vars_normal)}
    super(Laplace, self).initialize(var_list, *args, **kwargs)

  def finalize(self, feed_dict=None):
    """Function to call after convergence.

    Computes the Hessian at the mode.

    Parameters
    ----------
    feed_dict : dict, optional
      Feed dictionary for a TensorFlow session run during evaluation
      of Hessian. It is used to feed placeholders that are not fed
      during initialization.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    var_list = list(six.itervalues(self.latent_vars))
    hessians = tf.hessians(self.loss, var_list)

    assign_ops = []
    for z, hessian in zip(six.iterkeys(self.latent_vars), hessians):
      qz = self.latent_vars_normal[z]
      sigma_var = get_variables(qz.sigma)[0]
      if isinstance(qz, MultivariateNormalCholesky):
        sigma = tf.matrix_inverse(tf.cholesky(hessian))
      elif isinstance(qz, MultivariateNormalDiag):
        sigma = 1.0 / tf.diag_part(hessian)
      else:  # qz is MultivariateNormalFull
        sigma = tf.matrix_inverse(hessian)

      assign_ops.append(sigma_var.assign(sigma))

    sess = get_session()
    sess.run(assign_ops, feed_dict)
    self.latent_vars = self.latent_vars_normal.copy()
    del self.latent_vars_normal
    super(Laplace, self).finalize()
