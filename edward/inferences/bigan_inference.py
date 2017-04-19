from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.gan_inference import GANInference
from edward.util import get_session


class BiGANInference(GANInference):
  """Adversarially Learned Inference (Dumoulin et al., 2017) or
  Bidirectional Generative Adversarial Networks (Donahue et al., 2017)
  for joint learning of generator and inference networks.

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.
  """
  def __init__(self, latent_vars, data, discriminator):
    """
    Notes
    -----
    ``BiGANInference`` matches a mapping from data to latent variables and a
    mapping from latent variables to data through a joint
    discriminator.

    In building the computation graph for inference, the
    discriminator's parameters can be accessed with the variable scope
    "Disc".
    In building the computation graph for inference, the
    encoder and decoder parameters can be accessed with the variable scope
    "Gen".

    Examples
    --------
    >>> with tf.variable_scope("Gen"):
    >>>   xf = gen_data(z_ph)
    >>>   zf = gen_latent(x_ph)
    >>> inference = ed.BiGANInference({z_ph: zf}, {xf: x_ph}, discriminator)
    """
    if not callable(discriminator):
      raise TypeError("discriminator must be a callable function.")

    self.discriminator = discriminator
    # call grandparent's method; avoid parent (GANInference)
    super(GANInference, self).__init__(latent_vars, data)

  def build_loss_and_gradients(self, var_list):

    x_true = list(six.itervalues(self.data))[0]
    x_fake = list(six.iterkeys(self.data))[0]

    z_true = list(six.iterkeys(self.latent_vars))[0]
    z_fake = list(six.itervalues(self.latent_vars))[0]

    with tf.variable_scope("Disc"):
        # xtzf := x_true, z_fake
        d_xtzf = self.discriminator(x_true, z_fake)
    with tf.variable_scope("Disc", reuse=True):
        # xfzt := x_fake, z_true
        d_xfzt = self.discriminator(x_fake, z_true)

    loss_d = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_xfzt), logits=d_xfzt) + \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(d_xtzf), logits=d_xtzf)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_xfzt), logits=d_xfzt) + \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(d_xtzf), logits=d_xtzf)

    loss_d = tf.reduce_mean(loss_d)
    loss = tf.reduce_mean(loss)

    var_list_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gen")

    grads_d = tf.gradients(loss_d, var_list_d)
    grads = tf.gradients(loss, var_list)
    grads_and_vars_d = list(zip(grads_d, var_list_d))
    grads_and_vars = list(zip(grads, var_list))
    return loss, grads_and_vars, loss_d, grads_and_vars_d
