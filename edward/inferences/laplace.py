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
  def __init__(self, latent_vars, data=None, model_wrapper=None):
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
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("posterior"):
        if model_wrapper is None:
          latent_vars = {rv: MultivariateNormalCholesky(
              mu=tf.Variable(tf.random_normal(rv.batch_shape())),
              chol=tf.Variable(tf.random_normal(
                  rv.get_batch_shape().concatenate(rv.get_batch_shape()[-1]))))
              for rv in latent_vars}
        elif len(latent_vars) == 1:
          latent_vars = {latent_vars[0]: MultivariateNormalCholesky(
              mu=tf.Variable(tf.random_normal([model_wrapper.n_vars])),
              chol=tf.Variable(tf.random_normal([model_wrapper.n_vars] * 2)))}
        elif len(latent_vars) == 0:
          latent_vars = {}
        else:
          raise NotImplementedError("A list of more than one element is "
                                    "not supported. See documentation.")
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(
            qz, (MultivariateNormalCholesky, MultivariateNormalDiag,
                 MultivariateNormalFull)):
          raise TypeError("Posterior approximation must consist of only "
                          "MultivariateCholesky, MultivariateNormalDiag, "
                          "or MultivariateNormalFull random variables.")

    # call grandparent's method; avoid parent (MAP)
    super(MAP, self).__init__(latent_vars, data, model_wrapper)

  def initialize(self, var_list=None, *args, **kwargs):
    # TODO the algorithm relies on sampling from a pointmass
    # + maybe better to use pointmasses internally, where user passes
    # in normal distributions, we get its mean tf variables, and define
    # a pointmass with them.
    # + how to store it? map.py uses self.latent vars as the pointmass
    #   + for now, we can just hack it
    self.latent_vars_temp = self.latent_vars.copy()
    self.latent_vars = {z: PointMass(params=qz.mu)
                        for z, qz in six.iteritems(self.latent_vars_temp)}
    # # Variables may not be instantiated for model wrappers until
    # # their methods are first called. For now, hard-code
    # # ``var_list`` inside ``build_loss_and_gradients``.
    # if var_list is None and self.model_wrapper is None:
    #     # Traverse random variable graphs to get default list of variables.
    #     # For Laplace, the default is the mean parameters of the
    #     # normal approximation and any model parameters.
    #     var_list = set()
    #     trainables = tf.trainable_variables()
    #     for z, qz in six.iteritems(self.latent_vars):
    #       if isinstance(z, RandomVariable):
    #         var_list.update(get_variables(z, collection=trainables))

    #       var_list.update(get_variables(qz.mu, collection=trainables))

    #     for x, qx in six.iteritems(self.data):
    #       if isinstance(x, RandomVariable) and \
    #               not isinstance(qx, RandomVariable):
    #         var_list.update(get_variables(x, collection=trainables))

    #     var_list = list(var_list)
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
      qz = self.latent_vars_temp[z]
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
    self.latent_vars = self.latent_vars_temp.copy()
    del self.latent_vars_temp
    super(Laplace, self).finalize()
