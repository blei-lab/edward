from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import call_function_up_to_args


def gan_inference(model, discriminator, align_data,
                  collections=None, *args, **kwargs):
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

  `model` must return the generated data.
  """
  x_fake = call_function_up_to_args(model, *args, **kwargs)
  key = align_data(x_fake.name.split(':')[0])
  if isinstance(key, int):
    x_true = args[key]
  elif kwargs.get(key, None) is not None:
    x_true = kwargs.get(key)
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
  return loss, loss_d
