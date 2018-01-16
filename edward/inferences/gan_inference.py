from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import (check_and_maybe_build_data,
    transform, check_and_maybe_build_dict, check_and_maybe_build_var_list)


def gan_inference(data=None, discriminator=None,
                  scale=None, var_list=None, collections=None):
  """Parameter estimation with GAN-style training
  [@goodfellow2014generative].

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  #### Notes

  `GANInference` does not support latent variable inference. Note
  that GAN-style training also samples from the prior: this does not
  work well for latent variables that are shared across many data
  points (global variables).

  In building the computation graph for inference, the
  discriminator's parameters can be accessed with the variable scope
  "Disc".

  GANs also only work for one observed random variable in `data`.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  #### Examples

  ```python
  z = Normal(loc=tf.zeros([100, 10]), scale=tf.ones([100, 10]))
  x = generative_network(z)

  inference = ed.GANInference({x: x_data}, discriminator)
  ```
  """
  """Create an inference algorithm.

  Args:
    data: dict.
      Data dictionary which binds observed variables (of type
      `RandomVariable` or `tf.Tensor`) to their realizations (of
      type `tf.Tensor`).  It can also bind placeholders (of type
      `tf.Tensor`) used in the model to their realizations.
    discriminator: function.
      Function (with parameters) to discriminate samples. It should
      output logit probabilities (real-valued) and not probabilities
      in $[0, 1]$.
    var_list: list of tf.Variable, optional.
      List of TensorFlow variables to optimize over (in the generative
      model). Default is all trainable variables that `data` depends on.
  """
  if not callable(discriminator):
    raise TypeError("discriminator must be a callable function.")
  data = check_and_maybe_build_data(data)
  scale = check_and_maybe_build_dict(scale)
  var_list = check_and_maybe_build_var_list(var_list, {}, data)

  x_true = list(six.itervalues(data))[0]
  x_fake = list(six.iterkeys(data))[0]
  with tf.variable_scope("Disc"):
    d_true = discriminator(x_true)

  with tf.variable_scope("Disc", reuse=True):
    d_fake = discriminator(x_fake)

  if collections is not None:
    tf.summary.histogram("discriminator_outputs",
                         tf.concat([d_true, d_fake], axis=0),
                         collections=collections)

  reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
  reg_terms_all = tf.losses.get_regularization_losses()
  reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

  loss_d = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(d_true), logits=d_true) + \
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(d_fake), logits=d_fake)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(d_fake), logits=d_fake)
  loss_d = tf.reduce_mean(loss_d) + tf.reduce_sum(reg_terms_d)
  loss = tf.reduce_mean(loss) + tf.reduce_sum(reg_terms)

  var_list_d = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
  if var_list is None:
    var_list = [v for v in tf.trainable_variables() if v not in var_list_d]

  grads_d = tf.gradients(loss_d, var_list_d)
  grads = tf.gradients(loss, var_list)
  grads_and_vars_d = list(zip(grads_d, var_list_d))
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars, loss_d, grads_and_vars_d
