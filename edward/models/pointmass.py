"""The Point Mass distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.distributions.python.ops import \
    distribution
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

import tensorflow as tf


class PointMass(distribution.Distribution):
  def __init__(self,
               params,
               validate_args=True,
               allow_nan_stats=False,
               name="PointMass"):
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
    with ops.op_scope([params], name):
      params = ops.convert_to_tensor(params)
      with ops.control_dependencies([]):
        self._name = name
        self._params = array_ops.identity(params, name="params")
        self._batch_shape = self._ones().get_shape()
        self._event_shape = tensor_shape.TensorShape([])

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._params.dtype

  def batch_shape(self, name="batch_shape"):
    """Batch dimensions of this instance as a 1-D int32 `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op.

    Returns:
      `Tensor` `batch_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return array_ops.shape(self._ones())

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch shape
    """
    return self._batch_shape

  def event_shape(self, name="event_shape"):
    """Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op.

    Returns:
      `Tensor` `event_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return constant_op.constant([], dtype=dtypes.int32)

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event shape
    """
    return self._event_shape

  @property
  def params(self):
    """Distribution parameter."""
    return self._params

  def mean(self, name="mean"):
    """Mean of this distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._params], name):
        return self._params

  def mode(self, name="mode"):
    """Mode of this distribution."""
    return self.mean(name="mode")

  def std(self, name="std"):
    """Standard deviation of this distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._params], name):
        return 0.0 * array_ops.ones_like(self._params)

  def variance(self, name="variance"):
    """Variance of this distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return math_ops.square(self.std())

  def log_prob(self, x, name="log_prob"):
    """Log prob of observations in `x` under these Normal distribution(s).

    Args:
      x: tensor of dtype `dtype`, paramsst be broadcastable with `params`.
      name: The name to give this op.

    Returns:
      log_prob: tensor of dtype `dtype`, the log-PDFs of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._params, x], name):
        x = ops.convert_to_tensor(x)
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s"
                          % (x.dtype, self.dtype))
        return tf.cast(tf.equal(x, self.params), dtype=self.dtype)

  def cdf(self, x, name="cdf"):
    """CDF of observations in `x` under these Normal distribution(s).

    Args:
      x: tensor of dtype `dtype`, paramsst be broadcastable with `params`.
      name: The name to give this op.

    Returns:
      cdf: tensor of dtype `dtype`, the CDFs of `x`.
    """
    raise NotImplementedError()

  def log_cdf(self, x, name="log_cdf"):
    """Log CDF of observations `x` under these Normal distribution(s).

    Args:
      x: tensor of dtype `dtype`, paramsst be broadcastable with `params`.
      name: The name to give this op.

    Returns:
      log_cdf: tensor of dtype `dtype`, the log-CDFs of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._params, x], name):
        return math_ops.log(self.cdf(x))

  def prob(self, x, name="prob"):
    """The PDF of observations in `x` under these Normal distribution(s).

    Args:
      x: tensor of dtype `dtype`, paramsst be broadcastable with `params`.
      name: The name to give this op.

    Returns:
      prob: tensor of dtype `dtype`, the prob values of `x`.
    """
    return super(Normal, self).prob(x, name=name)

  def entropy(self, name="entropy"):
    """The entropy of Normal distribution(s).

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropy.
    """
    raise NotImplementedError()

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the Normal Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples for each
        of the distributions determined by broadcasting the hyperparameters.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._params, n], name):
        # TODO
        n = n.eval()
        return tf.pack([self.params] * n)

  @property
  def is_reparameterized(self):
    return True

  def _ones(self):
    return array_ops.ones_like(self._params)

  def _zeros(self):
    return array_ops.zeros_like(self._params)

  @property
  def is_continuous(self):
    return False
