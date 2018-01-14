from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import call_function_up_to_args


def wgan_inference(model, discriminator, align_data,
                   penalty=10.0, collections=None, *args, **kwargs):
  """Parameter estimation with GAN-style training
  [@goodfellow2014generative], using the Wasserstein distance
  [@arjovsky2017wasserstein].

  Works for the class of implicit (and differentiable) probabilistic
  models. These models do not require a tractable density and assume
  only a program that generates samples.

  The original WGAN clips weight parameters of the discriminator as an
  approximation to the 1-Lipschitz constraint. To clip weights, one
  must manually add a clipping op and then call it after each gradient
  update during training. For example:

  ```python
  ... = wgan_inference(data, discriminator, penalty=None)
  var_list = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope="Disc")
  clip_op = [w.assign(tf.clip_by_value(w, -0.1, 0.1)) for w in var_list]
  ```

  #### Notes

  Argument-wise, the only difference from `GANInference` is
  conceptual: the `discriminator` is better described as a test
  function or critic. `WGANInference` continues to use
  `discriminator` only to share methods and attributes with
  `GANInference`.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  #### Examples

  ```python
  z = Normal(loc=tf.zeros([100, 10]), scale=tf.ones([100, 10]))
  x = generative_network(z)

  inference = ed.WGANInference({x: x_data}, discriminator)
  ```
  """
  """Initialize inference algorithm. It initializes hyperparameters
  and builds ops for the algorithm's computation graph.

  Args:
    penalty: float, optional.
      Scalar value to enforce gradient penalty that ensures the
      gradients have norm equal to 1 [@gulrajani2017improved]. Set to
      None (or 0.0) if using no penalty.
    clip: float, optional.
      Value to clip weights by. Default is no clipping.

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
