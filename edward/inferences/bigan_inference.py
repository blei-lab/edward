from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import call_function_up_to_args
from edward.models.core import Trace


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_discriminator +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss_loss_d,
    notes_discriminator_scope=doc.notes_discriminator_scope,
    notes_regularization_losses=doc.notes_regularization_losses)
def bigan_inference(model, variational, discriminator, align_latent,
                    align_data, collections=None, *args, **kwargs):
  """Adversarially Learned Inference [@dumuolin2017adversarially] or
  Bidirectional Generative Adversarial Networks [@donahue2017adversarial]
  for joint learning of generator and inference networks.

  The function matches a mapping from data to latent variables and a
  mapping from latent variables to data through a joint discriminator.

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  Args:
  @{args}

  `align_latent` must only align one random variable in `model` and
  `variational`. `model` must return the generated data. `variational`
  assumes a random variable output and not an implicit density (or at
  least recorded on trace).

  Returns:
  @{returns}

  #### Notes

  @{notes_discriminator_scope}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    z = Normal(loc=0.0, scale=1.0, sample_shape=[256, 25], name="z")
    x = generative_network(z, name="x")
    return x

  def variational(x_data):
    net = tf.layers.dense(x_data, 25 * 2)
    qz = Normal(loc=net[:, :25],
                scale=tf.nn.softplus(net[:, 25:]),
                sample_shape=[256,],
                name="qz")

  def discriminator(x):
    net = tf.layers.dense(x, 256, activation=tf.nn.relu)
    return tf.layers.dense(net, 1, activation=tf.sigmoid)

  loss, loss_d = ed.bigan_inference(
      model, variational, discriminator,
      align_latent=lambda name: "qz" if name == "z" else None,
      align_data=lambda name: "x_data" if name == "x" else None,
      x_data=x_data)
  ```
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
