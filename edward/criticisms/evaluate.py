from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import RandomVariable
from edward.util import logit, get_session


def evaluate(metrics, data, n_samples=500, output_key=None):
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
    ``RandomVariable`` or ``tf.Tensor``) to their realizations (of
    type ``tf.Tensor``). It can also bind placeholders (of type
    ``tf.Tensor``) used in the model to their realizations.
  n_samples : int, optional
    Number of posterior samples for making predictions, using the
    posterior predictive distribution.
  output_key : RandomVariable, optional
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
  >>> # parameterized by a posterior sample
  >>> x_post = copy(x, {z: qz, beta: qbeta})
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
  feed_dict = {key: value for key, value in six.iteritems(data)
               if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type}

  if isinstance(metrics, str):
    metrics = [metrics]

  # Default output_key to the only data key that isn't a placeholder.
  if output_key is None:
    keys = [key for key in six.iterkeys(data) if not
            isinstance(key, tf.Tensor) or "Placeholder" not in key.op.type]
    if len(keys) == 1:
      output_key = keys[0]
    else:
      raise KeyError("User must specify output_key.")

  # Form true data. (It is required for all metrics excluding the
  # log-likelihood.)
  if metrics != ['log_lik'] and metrics != ['log_likelihood']:
    y_true = data[output_key]

  # Form predicted data (if there are any supervised metrics).
  if metrics != ['log_lik'] and metrics != ['log_likelihood']:
    # Monte Carlo estimate the mean of the posterior predictive.
    # Note the naive solution of taking the mean of
    # ``y_pred.sample(n_samples)`` does not work: ``y_pred`` is
    # parameterized by one posterior sample; this implies each
    # sample call from ``y_pred`` depends on the same posterior
    # sample. Instead, we fetch the sample tensor from the graph
    # many times. Alternatively, we could copy ``y_pred``
    # ``n_samples`` many times, so that each copy depends on a
    # different posterior sample. But it's expensive.
    tensor = tf.convert_to_tensor(output_key)
    y_pred = [sess.run(tensor, feed_dict) for _ in range(n_samples)]
    y_pred = tf.add_n(y_pred) / tf.cast(n_samples, tf.float32)

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
      # Monte Carlo estimate the log-density of the posterior predictive.
      tensor = tf.reduce_mean(output_key.log_prob(y_true))
      log_pred = [sess.run(tensor, feed_dict) for _ in range(n_samples)]
      log_pred = tf.add_n(log_pred) / tf.cast(n_samples, tf.float32)
      evaluations += [log_pred]
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
    Tensor of 0s and 1s (most generally, any real values a and b).
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
