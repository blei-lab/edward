from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.gan_inference import GANInference
from edward.util import get_session

try:
  from edward.models import Uniform
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class WGANInference(GANInference):
  """Parameter estimation with GAN-style training (Goodfellow et al.,
  2014), using the Wasserstein distance (Arjovsky et al., 2017).

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.
  """
  def __init__(self, *args, **kwargs):
    """
    Examples
    --------
    >>> z = Normal(loc=tf.zeros([100, 10]), scale=tf.ones([100, 10]))
    >>> x = generative_network(z)
    >>>
    >>> inference = ed.WGANInference({x: x_data}, discriminator)

    Notes
    -----
    Argument-wise, the only difference from ``GANInference`` is
    conceptual: the ``discriminator`` is better described as a test
    function or critic. ``WGANInference`` continues to use
    ``discriminator`` only to share methods and attributes with
    ``GANInference``.
    """
    super(WGANInference, self).__init__(*args, **kwargs)

  def initialize(self, penalty=10.0, clip=None, *args, **kwargs):
    """Initialize Wasserstein GAN inference.

    Parameters
    ----------
    penalty : float, optional
      Scalar value to enforce gradient penalty that ensures the
      gradients have norm equal to 1 (Gulrajani et al., 2017). Set to
      None (or 0.0) if using no penalty.
    clip : float, optional
      Value to clip weights by. Default is no clipping.
    """
    self.penalty = penalty

    super(WGANInference, self).initialize(*args, **kwargs)

    self.clip_op = None
    if clip is not None:
      var_list = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
      self.clip_op = [w.assign(tf.clip_by_value(w, -clip, clip))
                      for w in var_list]

  def build_loss_and_gradients(self, var_list):
    x_true = list(six.itervalues(self.data))[0]
    x_fake = list(six.iterkeys(self.data))[0]
    with tf.variable_scope("Disc"):
      d_true = self.discriminator(x_true)

    with tf.variable_scope("Disc", reuse=True):
      d_fake = self.discriminator(x_fake)

    if self.penalty is None or self.penalty == 0:
      penalty = 0.0
    else:
      eps = Uniform().sample(x_true.shape[0])
      while eps.shape.ndims < x_true.shape.ndims:
        eps = tf.expand_dims(eps, -1)
      x_interpolated = eps * x_true + (1.0 - eps) * x_fake
      with tf.variable_scope("Disc", reuse=True):
        d_interpolated = self.discriminator(x_interpolated)

      gradients = tf.gradients(d_interpolated, [x_interpolated])[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                     list(range(1, gradients.shape.ndims))))
      penalty = self.penalty * tf.reduce_mean(tf.square(slopes - 1.0))

    mean_true = tf.reduce_mean(d_true)
    mean_fake = tf.reduce_mean(d_fake)
    loss_d = mean_fake - mean_true + penalty
    loss = -mean_fake

    var_list_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    if var_list is None:
      var_list = [v for v in tf.trainable_variables() if v not in var_list_d]

    grads_d = tf.gradients(loss_d, var_list_d)
    grads = tf.gradients(loss, var_list)
    grads_and_vars_d = list(zip(grads_d, var_list_d))
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars, loss_d, grads_and_vars_d

  def update(self, feed_dict=None, variables=None):
    info_dict = super(WGANInference, self).update(feed_dict, variables)

    sess = get_session()
    if self.clip_op is not None and variables in (None, "Disc"):
      sess.run(self.clip_op)

    return info_dict
