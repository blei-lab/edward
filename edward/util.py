import numpy as np
import tensorflow as tf

def set_seed(x):
    """
    Set seed for both NumPy and TensorFlow.
    """
    np.random.seed(x)
    tf.set_random_seed(x)

def check_is_tf_vector(x):
    if isinstance(x, tf.Tensor):
        dimensions = x.get_shape()
        if(len(dimensions) == 0):
            raise TypeError("util::check_is_tf_vector: "
                            "input is a scalar.")
        elif(len(dimensions) == 1):
            if(dimensions[0].value <= 1):
                raise TypeError("util::check_is_tf_vector: "
                                "input has first dimension <= 1.")
            else:
                pass
        elif(len(dimensions) == 2):
            if(dimensions[1]!=1):
                raise TypeError("util::check_is_tf_vector: "
                                "input has second dimension != 1.")
        else:
            raise TypeError("util::check_is_tf_vector: "
                            "input has too many dimensions.")
    else:
        raise TypeError("util::check_is_tf_vector: "
                        "input is not a TensorFlow object.")

def log_sum_exp(x):
    """
    Computes the log_sum_exp of the elements in x.

    Works for x with
        shape=TensorShape([Dimension(N)])
        shape=TensorShape([Dimension(N), Dimension(1)])

    Not tested for anything beyond that.
    """
    check_is_tf_vector(x)
    x_max = tf.reduce_max(x)
    return tf.add(x_max, tf.log(tf.reduce_sum(tf.exp(tf.sub(x, x_max)))))

def logit(x):
    return tf.truediv(1.0, (1.0 + tf.exp(-x)))

def probit(x):
    return 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))

def sigmoid(x):
    "Numerically-stable sigmoid function."
    if x >= 0.0:
        z = tf.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = tf.exp(x)
        return z / (1.0 + z)

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

def trace(X):
    # assumes square
    n = X.get_shape()[0].value
    mask = tf.diag(tf.ones([n], dtype=X.dtype))
    X = tf.mul(mask, X)
    return tf.reduce_sum(X)

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

def kl_multivariate_normal(loc, scale):
    """
    KL( N(z; loc, scale) || N(z; 0, 1) ) for vector inputs, or
    sum_{m=1}^M KL( N(z_{m,:}; loc, scale) || N(z_{m,:}; 0, 1) ) for matrix inputs

    Parameters
    ----------
    loc : tf.Tensor
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the mean of a n-dimensional Gaussian
    scale : tf.Tensor
        n-dimensional vector, or M x n-dimensional matrix where each
        row represents the standard deviation of a n-dimensional Gaussian

    Returns
    -------
    tf.Tensor
        scalar
    """
    return -0.5 * tf.reduce_sum(1.0 + 2.0 * tf.log(scale + 1e-8) - \
                                tf.square(loc) - tf.square(scale))

def log_gamma(x):
    """
    TensorFlow doesn't have special functions, so use a
    log/exp/polynomial approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html
    """
    logterm = tf.log(x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log(xp3)

def log_beta(x, y):
    """
    TensorFlow doesn't have special functions, so use a
    log/exp/polynomial approximation.
    """
    return log_gamma(x) + log_gamma(y) - log_gamma(x+y)

def logit(x, clip_finite=True):
    if isinstance(x, tf.Tensor):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_logit_input")
        transformed = 1.0 / (1 + tf.exp(-x))
        jacobian = transformed * (1-transformed)
        if clip_finite:
            jacobian = tf.clip_by_value(jacobian, 1e-45, 1e38, name="clipped_jacobian")
        log_jacobian = tf.reduce_sum(tf.log(jacobian))

    else:
        transformed = 1.0 / (1 + np.exp(-x))
        jacobian = transformed * (1-transformed)
        log_jacobian = np.sum(np.log(jacobian))

    return transformed, log_jacobian

def multivariate_log_beta(x):
    return tf.reduce_sum(log_gamma(x)) - log_gamma(tf.reduce_sum(x))

def rbf(x):
    """RBF kernel element-wise."""
    return tf.exp(-0.5*x*x)

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
