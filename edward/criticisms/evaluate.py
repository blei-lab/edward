from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import RandomVariable
from edward.util import logit, get_session


def evaluate(metrics, data, latent_vars=None, model_wrapper=None,
             n_samples=100, output_key='y'):
  """Evaluate fitted model using a set of metrics.

  A metric, or scoring rule (Winkler, 1994), is a function of observed
  data under the posterior predictive distribution. For example in
  supervised metrics such as classification accuracy, the observed
  data (true output) is compared to the posterior predictive's mean
  (predicted output). In unsupervised metrics such as log-likelihood,
  the probability of observing the data is calculated under the
  posterior predictive's log-density.

  Parameters
  ----------
  metrics : list of str or str
    List of metrics or a single metric:
    ``'binary_accuracy'``,
    ``'categorical_accuracy'``,
    ``'sparse_categorical_accuracy'``,
    ``'log_loss'`` or ``'binary_crossentropy'``,
    ``'categorical_crossentropy'``,
    ``'sparse_categorical_crossentropy'``,
    ``'hinge'``,
    ``'squared_hinge'``,
    ``'mse'`` or ``'MSE'`` or ``'mean_squared_error'``,
    ``'mae'`` or ``'MAE'`` or ``'mean_absolute_error'``,
    ``'mape'`` or ``'MAPE'`` or ``'mean_absolute_percentage_error'``,
    ``'msle'`` or ``'MSLE'`` or ``'mean_squared_logarithmic_error'``,
    ``'poisson'``,
    ``'cosine'`` or ``'cosine_proximity'``,
    ``'log_lik'`` or ``'log_likelihood'``.
  data : dict
    Data to evaluate model with. It binds observed variables (of type
    ``RandomVariable``) to their realizations (of type ``tf.Tensor``). It
    can also bind placeholders (of type ``tf.Tensor``) used in the model
    to their realizations.
  latent_vars : dict of str to RandomVariable, optional
    Collection of random variables binded to their inferred posterior.
    It is only used (and in fact required) if the model wrapper is
    specified.
  model_wrapper : ed.Model, optional
    An optional wrapper for the probability model. It must have a
    ``predict`` method, and ``latent_vars`` must be specified. ``data`` is
    also changed. For TensorFlow, Python, and Stan models, the key
    type is a string; for PyMC3, the key type is a Theano shared
    variable. For TensorFlow, Python, and PyMC3 models, the value type
    is a NumPy array or TensorFlow placeholder; for Stan, the value
    type is the type according to the Stan program's data block.
  n_samples : int, optional
    Number of posterior samples for making predictions,
    using the posterior predictive distribution. It is only used if
    the model wrapper is specified.
  output_key : RandomVariable or str, optional
    It is the key in ``data`` which corresponds to the model's output.

  Returns
  -------
  list of float or float
    A list of evaluations or a single evaluation.

  Raises
  ------
  NotImplementedError
    If an input metric does not match an implemented metric in Edward.

  Examples
  --------
  >>> # build posterior predictive after inference: it is
  >>> # parameterized by posterior means
  >>> x_post = copy(x, {z: qz.mean(), beta: qbeta.mean()})
  >>>
  >>> # log-likelihood performance
  >>> evaluate('log_likelihood', data={x_post: x_train})
  >>>
  >>> # classification accuracy
  >>> # here, ``x_ph`` is any features the model is defined with respect to,
  >>> # and ``y_post`` is the posterior predictive distribution
  >>> evaluate('binary_accuracy', data={y_post: y_train, x_ph: x_train})
  >>>
  >>> # mean squared error
  >>> ed.evaluate('mean_squared_error', data={y: y_data, x: x_data})
  """
  sess = get_session()
  # Create feed_dict for data placeholders that the model conditions
  # on; it is necessary for all session runs.
  feed_dict = {x: obs for x, obs in six.iteritems(data)
               if not isinstance(x, RandomVariable) and
               not isinstance(x, str)}

  if isinstance(metrics, str):
    metrics = [metrics]

  # Set default for output_key if not using a model wrapper.
  if model_wrapper is None:
    if output_key == 'y':
      # Try to default to the only one observed random variable.
      keys = [key for key in six.iterkeys(data)
              if isinstance(key, RandomVariable)]
      if len(keys) == 1:
        output_key = keys[0]
      else:
        raise KeyError("User must specify output_key.")

  # Form true data. (It is not required in the specific setting of the
  # log-likelihood metric with a model wrapper.)
  if (metrics != ['log_lik'] and metrics != ['log_likelihood']) or \
          model_wrapper is None:
    y_true = data[output_key]

  # Form predicted data (if there are any supervised metrics).
  if metrics != ['log_lik'] and metrics != ['log_likelihood']:
    if model_wrapper is None:
      y_pred = output_key.mean()
    else:
      # Monte Carlo estimate the mean of the posterior predictive.
      y_pred = []
      for s in range(n_samples):
        zrep = {key: qz.sample(())
                for key, qz in six.iteritems(latent_vars)}
        y_pred += [model_wrapper.predict(data, zrep)]

      y_pred = tf.reduce_mean(y_pred, 0)

  # Evaluate y_true (according to y_pred if supervised) for all metrics.
  evaluations = []
  for metric in metrics:
    if metric == 'accuracy' or metric == 'crossentropy':
      # automate binary or sparse cat depending on its support
      support = tf.reduce_max(y_true).eval()
      if support <= 1:
        metric = 'binary_' + metric
      else:
        metric = 'sparse_categorical_' + metric

    if metric == 'binary_accuracy':
      evaluations += [binary_accuracy(y_true, y_pred)]
    elif metric == 'categorical_accuracy':
      evaluations += [categorical_accuracy(y_true, y_pred)]
    elif metric == 'sparse_categorical_accuracy':
      evaluations += [sparse_categorical_accuracy(y_true, y_pred)]
    elif metric == 'log_loss' or metric == 'binary_crossentropy':
      evaluations += [binary_crossentropy(y_true, y_pred)]
    elif metric == 'categorical_crossentropy':
      evaluations += [categorical_crossentropy(y_true, y_pred)]
    elif metric == 'sparse_categorical_crossentropy':
      evaluations += [sparse_categorical_crossentropy(y_true, y_pred)]
    elif metric == 'hinge':
      evaluations += [hinge(y_true, y_pred)]
    elif metric == 'squared_hinge':
      evaluations += [squared_hinge(y_true, y_pred)]
    elif (metric == 'mse' or metric == 'MSE' or
          metric == 'mean_squared_error'):
      evaluations += [mean_squared_error(y_true, y_pred)]
    elif (metric == 'mae' or metric == 'MAE' or
          metric == 'mean_absolute_error'):
      evaluations += [mean_absolute_error(y_true, y_pred)]
    elif (metric == 'mape' or metric == 'MAPE' or
          metric == 'mean_absolute_percentage_error'):
      evaluations += [mean_absolute_percentage_error(y_true, y_pred)]
    elif (metric == 'msle' or metric == 'MSLE' or
          metric == 'mean_squared_logarithmic_error'):
      evaluations += [mean_squared_logarithmic_error(y_true, y_pred)]
    elif metric == 'poisson':
      evaluations += [poisson(y_true, y_pred)]
    elif metric == 'cosine' or metric == 'cosine_proximity':
      evaluations += [cosine_proximity(y_true, y_pred)]
    elif metric == 'log_lik' or metric == 'log_likelihood':
      if model_wrapper is None:
        evaluations += [tf.reduce_mean(output_key.log_prob(y_true))]
      else:
        # Monte Carlo estimate the log-density of the posterior predictive.
        log_liks = []
        for s in range(n_samples):
          zrep = {key: qz.sample(())
                  for key, qz in six.iteritems(latent_vars)}
          log_liks += [model_wrapper.log_lik(data, zrep)]

        evaluations += [tf.reduce_mean(log_liks)]
    else:
      raise NotImplementedError()

  if len(evaluations) == 1:
    return sess.run(evaluations[0], feed_dict)
  else:
    return sess.run(evaluations, feed_dict)


# Classification metrics


def binary_accuracy(y_true, y_pred):
  """Binary prediction accuracy, also known as 0/1-loss.

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of 0s and 1s.
  y_pred : tf.Tensor
    Tensor of probabilities.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(tf.round(y_pred), tf.float32)
  return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def categorical_accuracy(y_true, y_pred):
  """Multi-class prediction accuracy. One-hot representation for ``y_true``.

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of 0s and 1s, where the outermost dimension of size ``K``
    has only one 1 per row.
  y_pred : tf.Tensor
    Tensor of probabilities, with same shape as ``y_true``.
    The outermost dimension denote the categorical probabilities for
    that data point per row.
  """
  y_true = tf.cast(tf.argmax(y_true, len(y_true.get_shape()) - 1), tf.float32)
  y_pred = tf.cast(tf.argmax(y_pred, len(y_pred.get_shape()) - 1), tf.float32)
  return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def sparse_categorical_accuracy(y_true, y_pred):
  """Multi-class prediction accuracy. Label {0, 1, .., K-1}
  representation for ``y_true``.

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of integers {0, 1, ..., K-1}.
  y_pred : tf.Tensor
    Tensor of probabilities, with shape ``(y_true.get_shape(), K)``.
    The outermost dimension are the categorical probabilities for
    that data point.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(tf.argmax(y_pred, len(y_pred.get_shape()) - 1), tf.float32)
  return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def binary_crossentropy(y_true, y_pred):
  """Binary cross-entropy.

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of 0s and 1s.
  y_pred : tf.Tensor
    Tensor of probabilities.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = logit(tf.cast(y_pred, tf.float32))
  return tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def categorical_crossentropy(y_true, y_pred):
  """Multi-class cross entropy. One-hot representation for ``y_true``.

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of 0s and 1s, where the outermost dimension of size K
    has only one 1 per row.
  y_pred : tf.Tensor
    Tensor of probabilities, with same shape as y_true.
    The outermost dimension denote the categorical probabilities for
    that data point per row.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = logit(tf.cast(y_pred, tf.float32))
  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def sparse_categorical_crossentropy(y_true, y_pred):
  """Multi-class cross entropy. Label {0, 1, .., K-1} representation
  for ``y_true.``

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of integers {0, 1, ..., K-1}.
  y_pred : tf.Tensor
    Tensor of probabilities, with shape ``(y_true.get_shape(), K)``.
    The outermost dimension are the categorical probabilities for
    that data point.
  """
  y_true = tf.cast(y_true, tf.int64)
  y_pred = logit(tf.cast(y_pred, tf.float32))
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=y_pred, labels=y_true))


def hinge(y_true, y_pred):
  """Hinge loss.

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of 0s and 1s.
  y_pred : tf.Tensor
    Tensor of real value.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.maximum(1.0 - y_true * y_pred, 0.0))


def squared_hinge(y_true, y_pred):
  """Squared hinge loss.

  Parameters
  ----------
  y_true : tf.Tensor
    Tensor of 0s and 1s.
  y_pred : tf.Tensor
    Tensor of real value.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.square(tf.maximum(1.0 - y_true * y_pred, 0.0)))


# Regression metrics


def mean_squared_error(y_true, y_pred):
  """Mean squared error loss.

  Parameters
  ----------
  y_true : tf.Tensor
  y_pred : tf.Tensor
    Tensors of same shape and type.
  """
  return tf.reduce_mean(tf.square(y_pred - y_true))


def mean_absolute_error(y_true, y_pred):
  """Mean absolute error loss.

  Parameters
  ----------
  y_true : tf.Tensor
  y_pred : tf.Tensor
    Tensors of same shape and type.
  """
  return tf.reduce_mean(tf.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
  """Mean absolute percentage error loss.

  Parameters
  ----------
  y_true : tf.Tensor
  y_pred : tf.Tensor
    Tensors of same shape and type.
  """
  diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true),
                                                     1e-8, np.inf))
  return 100.0 * tf.reduce_mean(diff)


def mean_squared_logarithmic_error(y_true, y_pred):
  """Mean squared logarithmic error loss.

  Parameters
  ----------
  y_true : tf.Tensor
  y_pred : tf.Tensor
    Tensors of same shape and type.
  """
  first_log = tf.log(tf.clip_by_value(y_pred, 1e-8, np.inf) + 1.0)
  second_log = tf.log(tf.clip_by_value(y_true, 1e-8, np.inf) + 1.0)
  return tf.reduce_mean(tf.square(first_log - second_log))


def poisson(y_true, y_pred):
  """Negative Poisson log-likelihood of data ``y_true`` given predictions
  ``y_pred`` (up to proportion).

  Parameters
  ----------
  y_true : tf.Tensor
  y_pred : tf.Tensor
    Tensors of same shape and type.
  """
  return tf.reduce_sum(y_pred - y_true * tf.log(y_pred + 1e-8))


def cosine_proximity(y_true, y_pred):
  """Cosine similarity of two vectors.

  Parameters
  ----------
  y_true : tf.Tensor
  y_pred : tf.Tensor
    Tensors of same shape and type.
  """
  y_true = tf.nn.l2_normalize(y_true, len(y_true.get_shape()) - 1)
  y_pred = tf.nn.l2_normalize(y_pred, len(y_pred.get_shape()) - 1)
  return tf.reduce_sum(y_true * y_pred)
