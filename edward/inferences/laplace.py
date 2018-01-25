from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.map import MAP
from edward.models import PointMass, RandomVariable
from edward.util import get_session, get_variables
from edward.util import copy, transform

try:
  from edward.models import \
      MultivariateNormalDiag, MultivariateNormalTriL, Normal
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class Laplace(MAP):
  """Laplace approximation [@laplace1986memoir].

  It approximates the posterior distribution using a multivariate
  normal distribution centered at the mode of the posterior.

  We implement this by running `MAP` to find the posterior mode.
  This forms the mean of the normal approximation. We then compute the
  inverse Hessian at the mode of the posterior. This forms the
  covariance of the normal approximation.

  #### Notes

  If `MultivariateNormalDiag` or `Normal` random variables are
  specified as approximations, then the Laplace approximation will
  only produce the diagonal. This does not capture correlation among
  the variables but it does not require a potentially expensive
  matrix inversion.

  Random variables with both scalar batch and event shape are not
  supported as `tf.hessians` is currently not applicable to scalars.

  Note that `Laplace` finds the location parameter of the normal
  approximation using `MAP`, which is performed on the latent
  variable's original (constrained) support. The scale parameter
  is calculated by evaluating the Hessian of $-\log p(x, z)$ in the
  constrained space and under the mode. This implies the Laplace
  approximation always has real support even if the target
  distribution has constrained support.

  #### Examples

  ```python
  X = tf.placeholder(tf.float32, [N, D])
  w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
  y = Normal(loc=ed.dot(X, w), scale=tf.ones(N))

  qw = MultivariateNormalTriL(
      loc=tf.Variable(tf.random_normal([D])),
      scale_tril=tf.Variable(tf.random_normal([D, D])))

  inference = ed.Laplace({w: qw}, data={X: X_train, y: y_train})
  ```
  """
  def __init__(self, latent_vars, data=None):
    """Create an inference algorithm.

    Args:
      latent_vars: list of RandomVariable or
                   dict of RandomVariable to RandomVariable.
        Collection of random variables to perform inference on. If list,
        each random variable will be implictly optimized using a
        `MultivariateNormalTriL` random variable that is defined
        internally with unconstrained support and is initialized using
        standard normal draws. If dictionary, each random
        variable must be a `MultivariateNormalDiag`,
        `MultivariateNormalTriL`, or `Normal` random variable.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope(None, default_name="posterior"):
        latent_vars_dict = {}
        for z in latent_vars:
          # Define location to have constrained support and
          # unconstrained free parameters.
          batch_event_shape = z.batch_shape.concatenate(z.event_shape)
          loc = tf.Variable(tf.random_normal(batch_event_shape))
          if hasattr(z, 'support'):
            z_transform = transform(z)
            if hasattr(z_transform, 'bijector'):
              loc = z_transform.bijector.inverse(loc)
          scale_tril = tf.Variable(tf.random_normal(
              batch_event_shape.concatenate(batch_event_shape[-1])))
          qz = MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
          latent_vars_dict[z] = qz
        latent_vars = latent_vars_dict
        del latent_vars_dict
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(
                qz, (MultivariateNormalDiag, MultivariateNormalTriL, Normal)):
          raise TypeError("Posterior approximation must consist of only "
                          "MultivariateNormalDiag, MultivariateTriL, or "
                          "Normal random variables.")

    # call grandparent's method; avoid parent (MAP)
    super(MAP, self).__init__(latent_vars, data)

  def initialize(self, *args, **kwargs):
    # Store latent variables in a temporary object; MAP will
    # optimize `PointMass` random variables, which subsequently
    # optimizes location parameters of the normal approximations.
    latent_vars_normal = self.latent_vars.copy()
    self.latent_vars = {z: PointMass(params=qz.loc)
                        for z, qz in six.iteritems(latent_vars_normal)}

    super(Laplace, self).initialize(*args, **kwargs)

    hessians = tf.hessians(self.loss, list(six.itervalues(self.latent_vars)))
    self.finalize_ops = []
    for z, hessian in zip(six.iterkeys(self.latent_vars), hessians):
      qz = latent_vars_normal[z]
      if isinstance(qz, (MultivariateNormalDiag, Normal)):
        scale_var = get_variables(qz.variance())[0]
        scale = 1.0 / tf.diag_part(hessian)
      else:  # qz is MultivariateNormalTriL
        scale_var = get_variables(qz.covariance())[0]
        scale = tf.matrix_inverse(tf.cholesky(hessian))

      self.finalize_ops.append(scale_var.assign(scale))

    self.latent_vars = latent_vars_normal.copy()
    del latent_vars_normal

  def finalize(self, feed_dict=None):
    """Function to call after convergence.

    Computes the Hessian at the mode.

    Args:
      feed_dict: dict.
        Feed dictionary for a TensorFlow session run during evaluation
        of Hessian. It is used to feed placeholders that are not fed
        during initialization.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    sess = get_session()
    sess.run(self.finalize_ops, feed_dict)
    super(Laplace, self).finalize()
