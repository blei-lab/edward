from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.inference import call_function_up_to_args


@doc.set_doc(
    args_part_one=(doc.arg_model +
                   doc.arg_discriminator +
                   doc.arg_align_data)[:-1],
    args_part_twoe=(doc.arg_collections +
                    doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss_loss_d,
    notes_discriminator_scope=doc.notes_discriminator_scope,
    notes_regularization_losses=doc.notes_regularization_losses)
def wgan_inference(model, discriminator, align_data,
                   penalty=10.0, collections=None, *args, **kwargs):
  """Parameter estimation with GAN-style training
  [@goodfellow2014generative], using the Wasserstein distance
  [@arjovsky2017wasserstein].

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  Args:
  @{args_part_one}
    penalty: float.
      Scalar value to enforce gradient penalty that ensures the
      gradients have norm equal to 1 [@gulrajani2017improved]. Set to
      None (or 0.0) if using no penalty.
  @{args_part_two}

  `model` must return the generated data.

  Returns:
  @{returns}

  #### Notes

  The original WGAN clips weight parameters of the discriminator as an
  approximation to the 1-Lipschitz constraint. To clip weights, one
  must manually add a clipping op and then call it after each gradient
  update during training. For example:

  ```python
  ... = wgan_inference(..., penalty=None)
  var_list = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
  clip_op = [w.assign(tf.clip_by_value(w, -0.1, 0.1)) for w in var_list]
  ```

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

  loss, loss_d = ed.wgan_inference(
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

  if penalty is None or penalty == 0:
    penalty = 0.0
  else:
    eps = tf.random_uniform(tf.shape(x_true))
    x_interpolated = eps * x_true + (1.0 - eps) * x_fake
    with tf.variable_scope("Disc", reuse=True):
      d_interpolated = discriminator(x_interpolated)

    gradients = tf.gradients(d_interpolated, [x_interpolated])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                   list(range(1, gradients.shape.ndims))))
    penalty = penalty * tf.reduce_mean(tf.square(slopes - 1.0))

  reg_terms_d = tf.losses.get_regularization_losses(scope="Disc")
  reg_terms_all = tf.losses.get_regularization_losses()
  reg_terms = [r for r in reg_terms_all if r not in reg_terms_d]

  mean_true = tf.reduce_mean(d_true)
  mean_fake = tf.reduce_mean(d_fake)
  loss_d = mean_fake - mean_true + penalty + tf.reduce_sum(reg_terms_d)
  loss = -mean_fake + tf.reduce_sum(reg_terms)
  return loss, loss_d
