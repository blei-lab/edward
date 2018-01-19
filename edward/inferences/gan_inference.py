from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.inference import call_function_up_to_args


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_discriminator +
          doc.arg_align_data +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss_loss_d,
    notes_discriminator_scope=doc.notes_discriminator_scope,
    notes_regularization_losses=doc.notes_regularization_losses)
def gan_inference(model, discriminator, align_data,
                  collections=None, *args, **kwargs):
  """Parameter estimation with GAN-style training
  [@goodfellow2014generative].

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  Args:
  @{args}

  `model` must return the generated data.

  Returns:
  @{returns}

  #### Notes

  @{notes_discriminator_scope}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    z = Normal(loc=0.0, scale=1.0, sample_shape=[256, 25])
    x = generative_network(z, name="x")
    return x

  def discriminator(x):
    net = tf.layers.dense(x, 256, activation=tf.nn.relu)
    return tf.layers.dense(net, 1, activation=tf.sigmoid)

  loss, loss_d = ed.gan_inference(
      model, discriminator,
      align_data=lambda name: "x_data" if name == "x" else None,
      x_data=x_data)
  ```
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
