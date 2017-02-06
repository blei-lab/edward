from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.util import get_session


class GANInference(VariationalInference):
  """Parameter estimation.

  Works for the class of implicit probabilistic models. Note that the
  algorithm samples from the prior: this does not work well for latent
  variables that are shared across many data points.
  """
  def __init__(self, latent_vars=None, data=None, discriminator=None):
    """
    Parameters
    ----------
    TODO latent_vars shouldn't exist, but it should exist as part of
    the inference to remain compatible?
    discriminator : function
      Function (with parameters) to discriminate samples.

    Does not work with model wrappers.
    """
    if discriminator is None:
      raise NotImplementedError()

    self.discriminator = discriminator
    super(GANInference, self).__init__(latent_vars, data, model_wrapper=None)

  # def initialize(self, *args, **kwargs):
  #   # TODO
  #   # 1. use one optimizer, sharing graph and optimizer from parent.
  #   # 2. use two separate optimizers.
  #   # + should we use generator optimizer in parent and discriminator optimizer here?
  #   # + this requires two separate functions. how to share pieces of the graph?
  #   # var_list = None
  #   # self.d_loss = self.build_discriminator_loss_and_gradients(var_list)
  #   return super(GANInference, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    # TODO
    # + var_list is ignored
    # + self.data for x and x_ph
    # + should probably use logits
    x = self.data.keys()[0]
    x_ph = self.data.values()[0]
    with tf.variable_scope("Disc"):
      p_true = self.discriminator(x_ph)

    with tf.variable_scope("Disc", reuse=True):
      p_fake = self.discriminator(x)

    loss_d = -tf.reduce_mean(tf.log(p_true) + tf.log(1 - p_fake))
    # TODO don't save it here?
    self.loss_d = loss_d
    var_list_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
    grads_d = tf.gradients(loss_d, var_list_d)

    loss_g = -tf.reduce_mean(tf.log(p_fake))
    var_list_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Gen")
    grads_g = tf.gradients(loss_g, var_list_g)

    loss = loss_g
    grads_and_vars = list(zip(grads_d, var_list_d) + zip(grads_g, var_list_g))
    return loss, grads_and_vars

  # def build_discriminator_loss_and_gradients(self, var_list):
  #   """Return grads and vars, which are the gradients for each
  #   parameter of the discriminator. The gradients are according to an
  #   auxiliary optimization problem.
  #   """
  #   return grads_and_vars

  def update(self, feed_dict=None):
    """Run one iteration of optimizer for variational inference.

    Parameters
    ----------
    feed_dict : dict, optional
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.

    Returns
    -------
    dict
      Dictionary of algorithm-specific information. In this case, the
      loss function value after one iteration.
    """
    if feed_dict is None:
      feed_dict = {}

    # TODO dealing with output that’s not a random variable but a tensor
    # + need to update the various contracts, and formalize this notion
    # for key, value in six.iteritems(self.data):
    #   if isinstance(key, tf.Tensor):
    #     feed_dict[key] = value

    sess = get_session()
    _, t, loss, loss_d = sess.run([self.train, self.increment_t, self.loss, self.loss_d], feed_dict)

    if self.debug:
      sess.run(self.op_check)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        if self.summarize is not None:
          summary = sess.run(self.summarize, feed_dict)
          self.train_writer.add_summary(summary, t)

    return {'t': t, 'loss': loss, 'loss_d': loss_d}

  def print_progress(self, info_dict):
    """Print progress to output.
    """
    if self.n_print != 0:
      t = info_dict['t']
      if t == 1 or t % self.n_print == 0:
        loss = info_dict['loss']
        loss_d = info_dict['loss_d']
        string = 'Iteration {0}'.format(str(t).rjust(len(str(self.n_iter))))
        string += ' [{0}%]'.format(str(int(t / self.n_iter * 100)).rjust(3))
        string += ': Gen Loss = {0:.3f}'.format(loss)
        string += ': Disc Loss = {0:.3f}'.format(loss_d)
        print(string)
