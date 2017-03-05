from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import warnings

from edward.util.random_variables import get_dims
from edward.util.graphs import get_session
from tensorflow.python.ops import control_flow_ops


def dot(x, y):
  """Compute dot product between a 2-D tensor and a 1-D tensor.

  If x is a ``[M x N]`` matrix, then y is a ``M``-vector.

  If x is a ``M``-vector, then y is a ``[M x N]`` matrix.

  Parameters
  ----------
  x : tf.Tensor
    A 1-D or 2-D tensor (see above).
  y : tf.Tensor
    A 1-D or 2-D tensor (see above).

  Returns
  -------
  tf.Tensor
    A 1-D tensor of length ``N``.

  Raises
  ------
  InvalidArgumentError
    If the inputs have Inf or NaN values.
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)
  dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg='')]
  x = control_flow_ops.with_dependencies(dependencies, x)
  y = control_flow_ops.with_dependencies(dependencies, y)

  if len(x.get_shape()) == 1:
    vec = x
    mat = y
    return tf.reshape(tf.matmul(tf.expand_dims(vec, 0), mat), [-1])
  else:
    mat = x
    vec = y
    return tf.reshape(tf.matmul(mat, tf.expand_dims(vec, 1)), [-1])


def hessian(y, xs):
  """Calculate Hessian of y with respect to each x in xs.

  Parameters
  ----------
  y : tf.Tensor
    Tensor to calculate Hessian of.
  xs : list of tf.Variable
    List of TensorFlow variables to calculate with respect to.
    The variables can have different shapes.

  Returns
  -------
  tf.Tensor
    A 2-D tensor where each row is
    .. math:: \partial_{xs} ( [ \partial_{xs} y ]_j ).

  Raises
  ------
  InvalidArgumentError
    If the inputs have Inf or NaN values.
  """
  y = tf.convert_to_tensor(y)
  dependencies = [tf.verify_tensor_all_finite(y, msg='')]
  dependencies.extend([tf.verify_tensor_all_finite(x, msg='') for x in xs])

  with tf.control_dependencies(dependencies):
    # Calculate flattened vector grad_{xs} y.
    grads = tf.gradients(y, xs)
    grads = [tf.reshape(grad, [-1]) for grad in grads]
    grads = tf.concat(grads, 0)
    # Loop over each element in the vector.
    mat = []
    d = grads.get_shape()[0]
    if not isinstance(d, int):
      d = grads.eval().shape[0]

    for j in range(d):
      # Calculate grad_{xs} ( [ grad_{xs} y ]_j ).
      gradjgrads = tf.gradients(grads[j], xs)
      # Flatten into vector.
      hi = []
      for l in range(len(xs)):
        hij = gradjgrads[l]
        # return 0 if gradient doesn't exist; TensorFlow returns None
        if hij is None:
          hij = tf.zeros(xs[l].get_shape(), dtype=tf.float32)

        hij = tf.reshape(hij, [-1])
        hi.append(hij)

      hi = tf.concat(hi, 0)
      mat.append(hi)

    # Form matrix where each row is grad_{xs} ( [ grad_{xs} y ]_j ).
    return tf.stack(mat)


def logit(x):
  """Evaluate :math:`\log(x / (1 - x))` elementwise.

  Parameters
  ----------
  x : tf.Tensor
    A n-D tensor.

  Returns
  -------
  tf.Tensor
    A tensor of same shape as input.

  Raises
  ------
  InvalidArgumentError
    If the input is not between :math:`(0,1)` elementwise.
  """
  x = tf.convert_to_tensor(x)
  dependencies = [tf.assert_positive(x),
                  tf.assert_less(x, 1.0)]
  x = control_flow_ops.with_dependencies(dependencies, x)

  return tf.log(x) - tf.log(1.0 - x)


def multivariate_rbf(x, y=0.0, sigma=1.0, l=1.0):
  """Squared-exponential kernel

  .. math:: k(x, y) = \sigma^2 \exp{ -1/(2l^2) \sum_i (x_i - y_i)^2 }

  Parameters
  ----------
  x : tf.Tensor
    A n-D tensor.
  y : tf.Tensor, optional
    A tensor of same shape as ``x``.
  sigma : tf.Tensor, optional
    A 0-D tensor, representing the standard deviation of radial
    basis function.
  l : tf.Tensor, optional
    A 0-D tensor, representing the lengthscale of radial basis
    function.

  Returns
  -------
  tf.Tensor
    A tensor of one less dimension than the input.

  Raises
  ------
  InvalidArgumentError
    If the mean variables have Inf or NaN values, or if the scale
    and length variables are not positive.
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)
  sigma = tf.convert_to_tensor(sigma)
  l = tf.convert_to_tensor(l)
  dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg=''),
                  tf.assert_positive(sigma),
                  tf.assert_positive(l)]
  x = control_flow_ops.with_dependencies(dependencies, x)
  y = control_flow_ops.with_dependencies(dependencies, y)
  sigma = control_flow_ops.with_dependencies(dependencies, sigma)
  l = control_flow_ops.with_dependencies(dependencies, l)

  return tf.pow(sigma, 2.0) * \
      tf.exp(-1.0 / (2.0 * tf.pow(l, 2.0)) * tf.reduce_sum(tf.pow(x - y, 2.0)))


def placeholder(*args, **kwargs):
  """A wrapper around ``tf.placeholder``. It adds the tensor to the
  ``PLACEHOLDERS`` collection."""
  warnings.simplefilter('default', DeprecationWarning)
  warnings.warn("ed.placeholder() is deprecated; use tf.placeholder() instead.",
                DeprecationWarning)
  x = tf.placeholder(*args, **kwargs)
  tf.add_to_collection("PLACEHOLDERS", x)
  return x


def rbf(x, y=0.0, sigma=1.0, l=1.0):
  """Squared-exponential kernel element-wise

  .. math:: k(x, y) = \sigma^2 \exp{ -1/(2l^2) (x - y)^2 }

  Parameters
  ----------
  x : tf.Tensor
    A n-D tensor.
  y : tf.Tensor, optional
    A tensor of same shape as ``x``.
  sigma : tf.Tensor, optional
    A 0-D tensor, representing the standard deviation of radial
    basis function.
  l : tf.Tensor, optional
    A 0-D tensor, representing the lengthscale of radial basis
    function.

  Returns
  -------
  tf.Tensor
    A tensor of one less dimension than the input.

  Raises
  ------
  InvalidArgumentError
    If the mean variables have Inf or NaN values, or if the scale
    and length variables are not positive.
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)
  sigma = tf.convert_to_tensor(sigma)
  l = tf.convert_to_tensor(l)
  dependencies = [tf.verify_tensor_all_finite(x, msg=''),
                  tf.verify_tensor_all_finite(y, msg=''),
                  tf.assert_positive(sigma),
                  tf.assert_positive(l)]
  x = control_flow_ops.with_dependencies(dependencies, x)
  y = control_flow_ops.with_dependencies(dependencies, y)
  sigma = control_flow_ops.with_dependencies(dependencies, sigma)
  l = control_flow_ops.with_dependencies(dependencies, l)

  return tf.pow(sigma, 2.0) * \
      tf.exp(-1.0 / (2.0 * tf.pow(l, 2.0)) * tf.pow(x - y, 2.0))


def reduce_logmeanexp(input_tensor, axis=None, keep_dims=False):
  """Computes log(mean(exp(elements across dimensions of a tensor))).

  Parameters
  ----------
  input_tensor : tf.Tensor
    The tensor to reduce. Should have numeric type.
  axis : int or list of int, optional
    The dimensions to reduce. If `None` (the default), reduces all
    dimensions.
  keep_dims : bool, optional
    If true, retains reduced dimensions with length 1.

  Returns
  -------
  tf.Tensor
    The reduced tensor.
  """
  logsumexp = tf.reduce_logsumexp(input_tensor, axis, keep_dims)
  input_tensor = tf.convert_to_tensor(input_tensor)
  n = input_tensor.get_shape().as_list()
  if axis is None:
    n = tf.cast(tf.reduce_prod(n), logsumexp.dtype)
  else:
    n = tf.cast(tf.reduce_prod(n[axis]), logsumexp.dtype)

  return -tf.log(n) + logsumexp


def to_simplex(x):
  """Transform real vector of length ``(K-1)`` to a simplex of dimension ``K``
  using a backward stick breaking construction.

  Parameters
  ----------
  x : tf.Tensor
    A 1-D or 2-D tensor.

  Returns
  -------
  tf.Tensor
    A tensor of same shape as input but with last dimension of
    size ``K``.

  Raises
  ------
  InvalidArgumentError
    If the input has Inf or NaN values.

  Notes
  -----
  x as a 3-D or higher tensor is not guaranteed to be supported.
  """
  x = tf.cast(x, dtype=tf.float32)
  dependencies = [tf.verify_tensor_all_finite(x, msg='')]
  x = control_flow_ops.with_dependencies(dependencies, x)

  if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
    shape = get_dims(x)
  else:
    shape = x.shape

  if len(shape) == 1:
    n_rows = ()
    K_minus_one = shape[0]
    eq = -tf.log(tf.cast(K_minus_one - tf.range(K_minus_one), dtype=tf.float32))
    z = tf.sigmoid(eq + x)
    pil = tf.concat([z, tf.constant([1.0])], 0)
    piu = tf.concat([tf.constant([1.0]), 1.0 - z], 0)
    S = tf.cumprod(piu)
    return S * pil
  else:
    n_rows = shape[0]
    K_minus_one = shape[1]
    eq = -tf.log(tf.cast(K_minus_one - tf.range(K_minus_one), dtype=tf.float32))
    z = tf.sigmoid(eq + x)
    pil = tf.concat([z, tf.ones([n_rows, 1])], 1)
    piu = tf.concat([tf.ones([n_rows, 1]), 1.0 - z], 1)
    S = tf.cumprod(piu, axis=1)
    return S * pil
