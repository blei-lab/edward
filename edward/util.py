import tensorflow as tf
import numpy as np

def cumprod(xs):
    """
    Cumulative product of a tensor along first dimension.
    https://github.com/tensorflow/tensorflow/issues/813
    """
    values = tf.unpack(xs)
    out = []
    prev = tf.ones_like(values[0])
    for val in values:
        s = prev * val
        out.append(s)
        prev = s

    result = tf.pack(out)
    return result

def digamma(x):
    """
    Computes the digamma function element-wise.

    TensorFlow doesn't have special functions with support for
    automatic differentiation, so use a log/exp/polynomial
    approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html

    Parameters
    ----------
    x : np.array or tf.Tensor
        scalar, vector, or rank-n tensor

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    """
    twopx = 2.0 + x
    logterm = tf.log(twopx)
    return - (1.0 + 2.0 * x) / (x * (1.0 + x)) - \
           (13.0 + 6.0 * x) / (12.0 * twopx * twopx) + logterm

def dot(x, y):
    """
    x is M x N matrix and y is N-vector, or
    x is M-vector and y is M x N matrix
    """
    if len(x.get_shape()) == 1:
        vec = x
        mat = y
        d = vec.get_shape()[0].value
        return tf.matmul(tf.reshape(vec, [1, d]), mat)
    else:
        mat = x
        vec = y
        d = vec.get_shape()[0].value
        return tf.matmul(mat, tf.reshape(vec, [d, 1]))

def get_dims(x):
    """
    Get values of each dimension.

    Arguments
    ----------
    x: tensor scalar or array
    """
    dims = x.get_shape()
    if len(dims) == 0: # scalar
        return [1]
    else: # array
        return [dim.value for dim in dims]

def kl_multivariate_normal(loc_one, scale_one, loc_two=0, scale_two=1):
    """
    Calculates the KL of multivariate normal distributions with
    diagonal covariances.

    Parameters
    ----------
    loc_one : tf.Tensor
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the mean of a n-dimensional Gaussian
    scale_one : tf.Tensor
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the standard deviation of a n-dimensional Gaussian
    loc_two : tf.Tensor, optional
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the mean of a n-dimensional Gaussian
    scale_two : tf.Tensor, optional
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the standard deviation of a n-dimensional Gaussian

    Returns
    -------
    tf.Tensor
        for scalar or vector inputs, outputs the scalar
            KL( N(z; loc_one, scale_one) || N(z; loc_two, scale_two) )
        for matrix inputs, outputs the vector
            [KL( N(z; loc_one[m,:], scale_one[m,:]) ||
                 N(z; loc_two[m,:], scale_two[m,:]) )]_{m=1}^M
    """
    if loc_two == 0 and scale_two == 1:
        return 0.5 * tf.reduce_sum(
            tf.square(scale_one) + tf.square(loc_one) - \
            1.0 - 2.0 * tf.log(scale_one))
    else:
        return 0.5 * tf.reduce_sum(
            tf.square(scale_one/scale_two) + \
            tf.square((loc_two - loc_one)/scale_two) - \
            1.0 + 2.0 * tf.log(scale_two) - 2.0 * tf.log(scale_one), 1)

def lbeta(x):
    """
    Computes the log of Beta(x), reducing along the last dimension.

    TensorFlow doesn't have special functions with support for
    automatic differentiation, so use a log/exp/polynomial
    approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html

    Parameters
    ----------
    x : np.array or tf.Tensor
        vector or rank-n tensor

    Returns
    -------
    tf.Tensor
        scalar if vector input, rank-(n-1) if rank-n tensor input
    """
    x = tf.cast(tf.squeeze(x), dtype=tf.float32)
    if len(get_dims(x)) == 1:
        return tf.reduce_sum(lgamma(x)) - lgamma(tf.reduce_sum(x))
    else:
        return tf.reduce_sum(lgamma(x), 1) - lgamma(tf.reduce_sum(x, 1))

def lgamma(x):
    """
    Computes the log of Gamma(x) element-wise.

    TensorFlow doesn't have special functions with support for
    automatic differentiation, so use a log/exp/polynomial
    approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html

    Parameters
    ----------
    x : np.array or tf.Tensor
        scalar, vector, or rank-n tensor

    Returns
    -------
    tf.Tensor
        size corresponding to size of input
    """
    logterm = tf.log(x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log(xp3)

def log_sum_exp(x):
    """
    Computes the log_sum_exp of the elements in x.

    Works for x with
        shape=TensorShape([Dimension(N)])
        shape=TensorShape([Dimension(N), Dimension(1)])

    Not tested for anything beyond that.
    """
    x_max = tf.reduce_max(x)
    return tf.add(x_max, tf.log(tf.reduce_sum(tf.exp(tf.sub(x, x_max)))))

def multivariate_rbf(x, y=0.0, sigma=1.0, l=1.0):
    """
    Squared-exponential kernel
    k(x, y) = sigma^2 exp{ -1/(2l^2) sum_i (x_i - y_i)^2 }
    """
    return tf.pow(sigma, 2.0) * \
           tf.exp(-1.0/(2.0*tf.pow(l, 2.0)) * \
                  tf.reduce_sum(tf.pow(x - y , 2.0)))

def rbf(x, y=0.0, sigma=1.0, l=1.0):
    """
    Squared-exponential kernel element-wise
    k(x, y) = sigma^2 exp{ -1/(2l^2) (x_i - y_i)^2 }
    """
    return tf.pow(sigma, 2.0) * \
           tf.exp(-1.0/(2.0*tf.pow(l, 2.0)) * tf.pow(x - y , 2.0))

def set_seed(x):
    """
    Set seed for both NumPy and TensorFlow.
    """
    np.random.seed(x)
    tf.set_random_seed(x)

# This is taken from PrettyTensor.
# https://github.com/google/prettytensor/blob/c9b69fade055d0eb35474fd23d07c43c892627bc/prettytensor/pretty_tensor_class.py#L1497
class VarStoreMethod(object):
  """Convenience base class for registered methods that create variables.
  This tracks the variables and requries subclasses to provide a __call__
  method.
  """

  def __init__(self):
    self.vars = {}

  def variable(self, var_name, shape, init=tf.random_normal_initializer(), dt=tf.float32, train=True):
    """Adds a named variable to this bookkeeper or returns an existing one.
    Variables marked train are returned by the training_variables method. If
    the requested name already exists and it is compatible (same shape, dt and
    train) then it is returned. In case of an incompatible type, an exception is
    thrown.
    Args:
      var_name: The unique name of this variable.  If a variable with the same
        name exists, then it is returned.
      shape: The shape of the variable.
      init: The init function to use or a Tensor to copy.
      dt: The datatype, defaults to float.  This will automatically extract the
        base dtype.
      train: Whether or not the variable should be trained.
    Returns:
      A TensorFlow tensor.
    Raises:
      ValueError: if reuse is False (or unspecified and allow_reuse is False)
        and the variable already exists or if the specification of a reused
        variable does not match the original.
    """
    # Make sure it is a TF dtype and convert it into a base dtype.
    dt = tf.as_dtype(dt).base_dtype
    if var_name in self.vars:
      v = self.vars[var_name]
      if v.get_shape() != shape:
        raise ValueError(
            'Shape mismatch: %s vs %s. Perhaps a UnboundVariable had '
            'incompatible values within a graph.' % (v.get_shape(), shape))
      return v
    elif callable(init):

      v = tf.get_variable(var_name,
                          shape=shape,
                          dtype=dt,
                          initializer=init,
                          trainable=train)
      self.vars[var_name] = v
      return v
    else:
      v = tf.convert_to_tensor(init, name=var_name, dtype=dt)
      v.get_shape().assert_is_compatible_with(shape)
      self.vars[var_name] = v
      return v

class VARIABLE(VarStoreMethod):
    """
    A simple wrapper to contain variables. It will create a TensorFlow
    variable the first time it is called and return the variable; in
    subsequent calls, it will simply return the variable and not
    create the TensorFlow variable again.

    This enables variables to be stored outside of classes which
    depend on parameters. It is a useful application for parametric
    distributions whose parameters may or may not be random (e.g.,
    through a prior), and for inverse mappings such as auto-encoders
    where we'd like to store inverse mapping parameters outside of the
    distribution class.
    """
    def __call__(self, name, shape):
        self.name = name
        return self.variable(name, shape)

Variable = VARIABLE()
