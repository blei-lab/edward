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


def kl_multivariate_normal(loc_one, scale_one, loc_two=0.0, scale_two=1.0):
  """Calculate the KL of multivariate normal distributions with
  diagonal covariances.

  Parameters
  ----------
  loc_one : tf.Tensor
    A 0-D tensor, 1-D tensor of length n, or 2-D tensor of shape M
    x n where each row represents the mean of a n-dimensional
    Gaussian.
  scale_one : tf.Tensor
    A tensor of same shape as ``loc_one``, representing the
    standard deviation.
  loc_two : tf.Tensor, optional
    A tensor of same shape as ``loc_one``, representing the
    mean of another Gaussian.
  scale_two : tf.Tensor, optional
    A tensor of same shape as ``loc_one``, representing the
    standard deviation of another Gaussian.

  Returns
  -------
  tf.Tensor
    For 0-D or 1-D tensor inputs, outputs the 0-D tensor
    ``KL( N(z; loc_one, scale_one) || N(z; loc_two, scale_two) )``
    For 2-D tensor inputs, outputs the 1-D tensor
    ``[KL( N(z; loc_one[m,:], scale_one[m,:]) || ``
    ``N(z; loc_two[m,:], scale_two[m,:]) )]_{m=1}^M``

  Raises
  ------
  InvalidArgumentError
    If the location variables have Inf or NaN values, or if the scale
    variables are not positive.
  """
  loc_one = tf.convert_to_tensor(loc_one)
  scale_one = tf.convert_to_tensor(scale_one)
  loc_two = tf.convert_to_tensor(loc_two)
  scale_two = tf.convert_to_tensor(scale_two)
  dependencies = [tf.verify_tensor_all_finite(loc_one, msg=''),
                  tf.verify_tensor_all_finite(loc_two, msg=''),
                  tf.assert_positive(scale_one),
                  tf.assert_positive(scale_two)]
  loc_one = control_flow_ops.with_dependencies(dependencies, loc_one)
  scale_one = control_flow_ops.with_dependencies(dependencies, scale_one)

  if loc_two == 0.0 and scale_two == 1.0:
    # With default arguments, we can avoid some intermediate computation.
    out = tf.square(scale_one) + tf.square(loc_one) - \
        1.0 - 2.0 * tf.log(scale_one)
  else:
    loc_two = control_flow_ops.with_dependencies(dependencies, loc_two)
    scale_two = control_flow_ops.with_dependencies(dependencies, scale_two)
    out = tf.square(scale_one / scale_two) + \
        tf.square((loc_two - loc_one) / scale_two) - \
        1.0 + 2.0 * tf.log(scale_two) - 2.0 * tf.log(scale_one)

  if len(out.get_shape()) <= 1:  # scalar or vector
    return 0.5 * tf.reduce_sum(out)
  else:  # matrix
    return 0.5 * tf.reduce_sum(out, 1)


def log_mean_exp(input_tensor, axis=None, keep_dims=False):
  """Compute the ``log_mean_exp`` of elements in a tensor, taking
  the mean across axes given by ``axis``.

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

  Raises
  ------
  InvalidArgumentError
    If the input has Inf or NaN values.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  dependencies = [tf.verify_tensor_all_finite(input_tensor, msg='')]
  input_tensor = control_flow_ops.with_dependencies(dependencies, input_tensor)

  x_max = tf.reduce_max(input_tensor, axis, keep_dims=True)
  return tf.squeeze(x_max) + tf.log(tf.reduce_mean(
      tf.exp(input_tensor - x_max), axis, keep_dims))


def log_sum_exp(input_tensor, axis=None, keep_dims=False):
  """Compute the ``log_sum_exp`` of elements in a tensor, taking
  the sum across axes given by ``axis``.

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

  Raises
  ------
  InvalidArgumentError
    If the input has Inf or NaN values.
  """
  input_tensor = tf.convert_to_tensor(input_tensor)
  dependencies = [tf.verify_tensor_all_finite(input_tensor, msg='')]
  input_tensor = control_flow_ops.with_dependencies(dependencies, input_tensor)

  x_max = tf.reduce_max(input_tensor, axis, keep_dims=True)
  return tf.squeeze(x_max) + tf.log(tf.reduce_sum(
      tf.exp(input_tensor - x_max), axis, keep_dims))


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


def tile(input, multiples, *args, **kwargs):
  """Constructs a tensor by tiling a given tensor.

  This extends ``tf.tile`` to features available in ``np.tile``.
  Namely, ``inputs`` and ``multiples`` can be a 0-D tensor.  Further,
  if 1-D, ``multiples`` can be of any length according to broadcasting
  rules (see documentation of ``np.tile`` or examples below).

  Parameters
  ----------
  input : tf.Tensor
    The input tensor.
  multiples : tf.Tensor
    The number of repetitions of ``input`` along each axis. Has type
    ``tf.int32``. 0-D or 1-D.
  *args :
    Passed into ``tf.tile``.
  **kwargs :
    Passed into ``tf.tile``.

  Returns
  -------
  tf.Tensor
      Has the same type as ``input``.

  Examples
  --------
  >>> a = tf.constant([0, 1, 2])
  >>> sess.run(ed.tile(a, 2))
  array([0, 1, 2, 0, 1, 2], dtype=int32)
  >>> sess.run(ed.tile(a, (2, 2)))
  array([[0, 1, 2, 0, 1, 2],
         [0, 1, 2, 0, 1, 2]], dtype=int32)
  >>> sess.run(ed.tile(a, (2, 1, 2)))
  array([[[0, 1, 2, 0, 1, 2]],
         [[0, 1, 2, 0, 1, 2]]], dtype=int32)
  >>>
  >>> b = tf.constant([[1, 2], [3, 4]])
  >>> sess.run(ed.tile(b, 2))
  array([[1, 2, 1, 2],
         [3, 4, 3, 4]], dtype=int32)
  >>> sess.run(ed.tile(b, (2, 1)))
  array([[1, 2],
         [3, 4],
         [1, 2],
         [3, 4]], dtype=int32)
  >>>
  >>> c = tf.constant([1, 2, 3, 4])
  >>> sess.run(ed.tile(c, (4, 1)))
  array([[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]], dtype=int32)

  Notes
  -----
  Sometimes this can result in an unknown shape. The core reason for
  this is the following behavior:

  >>> n = tf.constant([1])
  >>> tf.tile(tf.constant([[1.0]]),
  ...         tf.concat([n, tf.constant([1.0]).get_shape()]), 0)
  <tf.Tensor 'Tile:0' shape=(1, 1) dtype=float32>
  >>> n = tf.reshape(tf.constant(1), [1])
  >>> tf.tile(tf.constant([[1.0]]),
  ...         tf.concat([n, tf.constant([1.0]).get_shape()]), 0)
  <tf.Tensor 'Tile_1:0' shape=(?, 1) dtype=float32>

  For this reason, we try to fetch ``multiples`` out of session if
  possible. This can be slow if ``multiples`` has computationally
  intensive dependencies in order to perform this fetch.
  """
  input = tf.convert_to_tensor(input)
  multiples = tf.convert_to_tensor(multiples)

  # 0-d tensor
  if len(input.get_shape()) == 0:
    input = tf.expand_dims(input, 0)

  # 0-d tensor
  if len(multiples.get_shape()) == 0:
    multiples = tf.expand_dims(multiples, 0)

  try:
    get_session()
    multiples = tf.convert_to_tensor(multiples.eval())
  except:
    pass

  # broadcasting
  diff = len(input.get_shape()) - get_dims(multiples)[0]
  if diff < 0:
    input = tf.reshape(input, [1] * np.abs(diff) + get_dims(input))
  elif diff > 0:
    multiples = tf.concat([tf.ones(diff, dtype=tf.int32), multiples], 0)

  return tf.tile(input, multiples, *args, **kwargs)


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
