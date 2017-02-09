from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.gan_inference import GANInference
from edward.util import get_session


class AVB(GANInference):
  """Adversarial variational Bayes (Mescheder et al., 2017).
  """
  def __init__(self, latent_vars, data=None, discriminator=None):
    """
    Parameters
    ----------
    discriminator : function
      Function (with parameters) to discriminate samples. It should
      output logit probabilities (real-valued) and not probabilities
      in [0, 1]. Unlike ``GANInference``, it takes two arguments: a
      data tensor and latent variable tensor.

    Notes
    -----
    Adversarial variational Bayes only infers one latent variable, so
    ``latent_vars`` must be of length 1. Similarly, ``data`` can have
    only one observed random variable.

    Because it requires sampling from the prior, ``AVB`` will not work
    well for inferring latent variables shared across data points
    (e.g., coefficients in linear regression). Also, unlike
    ``GANInference``, it requires that the likelihood is tractable.

    Examples
    --------
    >>> z = Normal(mu=tf.zeros([100, 10]), sigma=tf.ones([100, 10]))
    >>> x = Normal(mu=generative_network(z), sigma=1.0)
    >>>
    >>> qz = Normal(mu=tf.Variable([100, 10]),
    >>>             sigma=tf.nn.softplus(tf.Variable([100, 10])))
    >>>
    >>> inference = ed.AVB({z: qz}, {x: x_data}, discriminator)
    """
    if discriminator is None:
      raise NotImplementedError()

    self.discriminator = discriminator
    # call grandparent's method; avoid parent (GANInference)
    super(GANInference, self).__init__(latent_vars, data, model_wrapper=None)

  def build_loss_and_gradients(self, var_list):
    # Collect tensors used in calculation of losses.
    pz = list(six.iterkeys(self.latent_vars))[0]
    qz = list(six.itervalues(self.latent_vars))[0]
    pz_sample = pz.sample()
    qz_sample = qz.sample()

    p_log_lik = 0.0  # instantiate in case there is no data
    x_true = None
    for x, x_data in six.iterkeys(self.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap={pz: qz_sample},
                      scope='inference_' + str(id(self)))
        p_log_lik = tf.reduce_sum(x_copy.log_lik(x_data))
        x_true = x_data

    with tf.variable_scope("Disc"):
      d_true = self.discriminator(x_true, qz_sample)

    with tf.variable_scope("Disc", reuse=True):
      d_fake = self.discriminator(x_true, pz_sample)

    # Form variational objective and auxiliary log-ratio loss.
    loss = -tf.reduce_mean(p_log_lik - d_true)
    loss_d = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_true), logits=d_true) + \
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(d_fake), logits=d_fake)
    loss_d = tf.reduce_mean(loss_d)

    var_list_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    if var_list is None:
      var_list = [v for v in tf.trainable_variables() if v not in var_list_d]

    grads = tf.gradients(loss, var_list)
    grads_d = tf.gradients(loss_d, var_list_d)
    grads_and_vars = list(zip(grads, var_list))
    grads_and_vars_d = list(zip(grads_d, var_list_d))
    return loss, grads_and_vars, loss_d, grads_and_vars_d
