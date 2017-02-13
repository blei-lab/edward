from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.gan_inference import GANInference
from edward.util import get_session


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
    >>> z = Normal(mu=tf.zeros([100, 10]), sigma=tf.ones([100, 10]))
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

  def initialize(self, *args, **kwargs):
    super(WGANInference, self).initialize(*args, **kwargs)

    var_list = kwargs.get('var_list', None)
    var_list_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    if var_list is not None:
      var_list_d = list(set(var_list_d) & set(var_list))

    clip_d = [w.assign(tf.clip_by_value(w, -0.01, 0.01))
              for w in var_list_d]
    self.clip_d = clip_d

  def build_loss_and_gradients(self, var_list):
    x_true = list(six.itervalues(self.data))[0]
    x_fake = list(six.iterkeys(self.data))[0]
    with tf.variable_scope("Disc"):
      d_true = self.discriminator(x_true)

    with tf.variable_scope("Disc", reuse=True):
      d_fake = self.discriminator(x_fake)

    mean_true = tf.reduce_mean(d_true)
    mean_fake = tf.reduce_mean(d_fake)
    loss_d = -mean_true + mean_fake
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
    if variables is None or variables == "Disc":
      sess.run(self.clip_d)

    return info_dict
