from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import call_function_up_to_args
from edward.models.core import Trace


def bigan_inference(model, variational, discriminator, align_data,
                    align_latent, collections=None, *args, **kwargs):
  """Adversarially Learned Inference [@dumuolin2017adversarially] or
  Bidirectional Generative Adversarial Networks [@donahue2017adversarial]
  for joint learning of generator and inference networks.

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  #### Notes

  `BiGANInference` matches a mapping from data to latent variables and a
  mapping from latent variables to data through a joint
  discriminator.

  In building the computation graph for inference, the
  discriminator's parameters can be accessed with the variable scope
  "Disc".
  In building the computation graph for inference, the
  encoder and decoder parameters can be accessed with the variable scope
  "Gen".

  The objective function also adds to itself a summation over all tensors
  in the `REGULARIZATION_LOSSES` collection.

  #### Examples

  ```python
  with tf.variable_scope("Gen"):
    xf = gen_data(z_ph)
    zf = gen_latent(x_ph)
  inference = ed.BiGANInference({z_ph: zf}, {xf: x_ph}, discriminator)
  ```

  `align_latent` must only align one random variable in `model` and
  `variational`. `model` must return the generated data.
  """
  with Trace() as posterior_trace:
    call_function_up_to_args(variational, *args, **kwargs)
  with Trace() as model_trace:
    x_fake = call_function_up_to_args(model, *args, **kwargs)

  key = align_data(x_fake.name.split(':')[0])
  if isinstance(key, int):
    x_true = args[key]
  elif kwargs.get(key, None) is not None:
    x_true = kwargs.get(key)

  for name, node in six.iteritems(model_trace):
    aligned = align_latent(name)
    if aligned is not None:
      z_true = node.value
      z_fake = posterior_trace[aligned].value
      break

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

  reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
  reg_terms_all = tf.losses.get_regularization_losses()
  reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

  loss_d = tf.reduce_mean(loss_d) + tf.reduce_sum(reg_terms_d)
  loss = tf.reduce_mean(loss) + tf.reduce_sum(reg_terms)
  return loss, loss_d
