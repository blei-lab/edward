from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import RandomVariable
from edward.util import check_data, get_session, compute_multinomial_mode, \
    with_binary_averaging

try:
  from edward.models import Bernoulli, Binomial, Categorical, \
      Multinomial, OneHotCategorical
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def evaluate(metrics, data, n_samples=500, output_key=None, seed=None):
  """Evaluate fitted model using a set of metrics.

  A metric, or scoring rule [@winkler1994evaluating], is a function of
  observed data under the posterior predictive distribution. For
  example in supervised metrics such as classification accuracy, the
  observed data (true output) is compared to the posterior
  predictive's mean (predicted output). In unsupervised metrics such
  as log-likelihood, the probability of observing the data is
  calculated under the posterior predictive's log-density.

  Args:
    metrics: list of str and/or (str, params: dict) tuples, str,
    or (str, params: dict) tuple.
      List of metrics or a single metric:
      `'binary_accuracy'`,
      `'categorical_accuracy'`,
      `'sparse_categorical_accuracy'`,
      `'log_loss'` or `'binary_crossentropy'`,
      `'categorical_crossentropy'`,
      `'sparse_categorical_crossentropy'`,
      `'hinge'`,
      `'squared_hinge'`,
      `'mse'` or `'MSE'` or `'mean_squared_error'`,
      `'mae'` or `'MAE'` or `'mean_absolute_error'`,
      `'mape'` or `'MAPE'` or `'mean_absolute_percentage_error'`,
      `'msle'` or `'MSLE'` or `'mean_squared_logarithmic_error'`,
      `'poisson'`,
      `'cosine'` or `'cosine_proximity'`,
      `'log_lik'` or `'log_likelihood'`.
      In lieu of a metric string, this method also accepts (str, params: dict)
      tuples; the first element of this tuple is the metric string, and
      the second is a dict of associated params. At present, this dict only
      expects one key, `'average'`, which stipulates the type of averaging to
      perform on those metrics that permit binary averaging. Permissible
      options include: `None`, `'macro'` and `'micro'`.
    data: dict.
      Data to evaluate model with. It binds observed variables (of type
      `RandomVariable` or `tf.Tensor`) to their realizations (of
      type `tf.Tensor`). It can also bind placeholders (of type
      `tf.Tensor`) used in the model to their realizations.
    n_samples: int.
      Number of posterior samples for making predictions, using the
      posterior predictive distribution.
    output_key: RandomVariable or tf.Tensor.
      It is the key in `data` which corresponds to the model's output.
    seed: a Python integer. Used to create a random seed for the
      distribution

  Returns:
    list of float or float.
    A list of evaluations or a single evaluation.

  Raises:
    NotImplementedError.
    If an input metric does not match an implemented metric in Edward.

  #### Examples

  ```python
  # build posterior predictive after inference: it is
  # parameterized by a posterior sample
  x_post = ed.copy(x, {z: qz, beta: qbeta})

  # log-likelihood performance
  ed.evaluate('log_likelihood', data={x_post: x_train})

  # classification accuracy
  # here, `x_ph` is any features the model is defined with respect to,
  # and `y_post` is the posterior predictive distribution
  ed.evaluate('binary_accuracy', data={y_post: y_train, x_ph: x_train})

  # mean squared error
  ed.evaluate('mean_squared_error', data={y: y_data, x: x_data})
  ```

  # mean squared logarithmic error with `'micro'` averaging
  ed.evaluate(('mean_squared_logarithmic_error', {'average': 'micro'}),
              data={y: y_data, x: x_data})
  """
  sess = get_session()
  if isinstance(metrics, str):
    metrics = [metrics]
  elif callable(metrics):
    metrics = [metrics]
  elif not isinstance(metrics, list):
    raise TypeError("metrics must have type str or list, or be callable.")

  check_data(data)
  if not isinstance(n_samples, int):
    raise TypeError("n_samples must have type int.")

  if output_key is None:
    # Default output_key to the only data key that isn't a placeholder.
    keys = [key for key in six.iterkeys(data) if not
            isinstance(key, tf.Tensor) or "Placeholder" not in key.op.type]
    if len(keys) == 1:
      output_key = keys[0]
    else:
      raise KeyError("User must specify output_key.")
  elif not isinstance(output_key, RandomVariable):
    raise TypeError("output_key must have type RandomVariable.")

  # Create feed_dict for data placeholders that the model conditions
  # on; it is necessary for all session runs.
  feed_dict = {key: value for key, value in six.iteritems(data)
               if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type}

  # Form true data.
  y_true = data[output_key]
  # Make predictions (if there are any supervised metrics).
  if metrics != ['log_lik'] and metrics != ['log_likelihood']:
    binary_discrete = (Bernoulli, Binomial)
    categorical_discrete = (Categorical, Multinomial, OneHotCategorical)
    total_count = sess.run(getattr(output_key, 'total_count', tf.constant(1.)))
    if isinstance(output_key, binary_discrete + categorical_discrete):
      # Average over realizations of their probabilities, then predict
      # via argmax over probabilities.
      probs = [sess.run(output_key.probs, feed_dict) for _ in range(n_samples)]
      probs = np.sum(probs, axis=0) / n_samples
      if isinstance(output_key, binary_discrete):
        # make random prediction whenever probs is exactly 0.5
        random = tf.random_uniform(shape=tf.shape(probs))
        y_pred = tf.round(tf.where(tf.equal(0.5, probs), random, probs))
      else:
        if total_count > 1:
          mode = compute_multinomial_mode(probs, total_count, seed)
          if len(output_key.sample_shape):
            y_pred = tf.reshape(tf.tile(mode, output_key.sample_shape),
                                [-1, len(probs)])
          else:
            y_pred = mode
        else:
          y_pred = tf.argmax(probs, len(probs.shape) - 1)
      probs = tf.constant(probs)
    else:
      # Monte Carlo estimate the mean of the posterior predictive.
      y_pred = [sess.run(output_key, feed_dict) for _ in range(n_samples)]
      y_pred = tf.cast(tf.add_n(y_pred), y_pred[0].dtype) / \
          tf.cast(n_samples, y_pred[0].dtype)
    if len(y_true.shape) == 0:
      y_true = tf.expand_dims(y_true, 0)
      y_pred = tf.expand_dims(y_pred, 0)

  # Evaluate y_true (according to y_pred if supervised) for all metrics.
  evaluations = []
  for metric in metrics:
    if isinstance(metric, tuple):
      metric, params = metric
    else:
      params = {}
    if metric == 'accuracy' or metric == 'crossentropy':
      # automate binary or sparse cat depending on its support
      support = sess.run(tf.reduce_max(y_true), feed_dict)
      if support <= 1:
        metric = 'binary_' + metric
      else:
        metric = 'sparse_categorical_' + metric

    if metric == 'binary_accuracy':
      evaluations += [binary_accuracy(y_true, y_pred, **params)]
    elif metric == 'categorical_accuracy':
      evaluations += [categorical_accuracy(y_true, y_pred, **params)]
    elif metric == 'sparse_categorical_accuracy':
      evaluations += [sparse_categorical_accuracy(y_true, y_pred, **params)]
    elif metric == 'log_loss' or metric == 'binary_crossentropy':
      evaluations += [binary_crossentropy(y_true, y_pred, **params)]
    elif metric == 'categorical_crossentropy':
      evaluations += [categorical_crossentropy(y_true, y_pred, **params)]
    elif metric == 'sparse_categorical_crossentropy':
      evaluations += [sparse_categorical_crossentropy(y_true, y_pred, **params)]
    elif metric == 'multinomial_accuracy':
      evaluations += [multinomial_accuracy(y_true, y_pred, **params)]
    elif metric == 'kl_divergence':
      y_true_ = y_true / total_count
      y_pred_ = probs
      evaluations += [kl_divergence(y_true_, y_pred_, **params)]
    elif metric == 'hinge':
      evaluations += [hinge(y_true, y_pred, **params)]
    elif metric == 'squared_hinge':
      evaluations += [squared_hinge(y_true, y_pred, **params)]
    elif (metric == 'mse' or metric == 'MSE' or
          metric == 'mean_squared_error'):
      evaluations += [mean_squared_error(y_true, y_pred, **params)]
    elif (metric == 'mae' or metric == 'MAE' or
          metric == 'mean_absolute_error'):
      evaluations += [mean_absolute_error(y_true, y_pred, **params)]
    elif (metric == 'mape' or metric == 'MAPE' or
          metric == 'mean_absolute_percentage_error'):
      evaluations += [mean_absolute_percentage_error(y_true, y_pred, **params)]
    elif (metric == 'msle' or metric == 'MSLE' or
          metric == 'mean_squared_logarithmic_error'):
      evaluations += [mean_squared_logarithmic_error(y_true, y_pred, **params)]
    elif metric == 'poisson':
      evaluations += [poisson(y_true, y_pred, **params)]
    elif metric == 'cosine' or metric == 'cosine_proximity':
      evaluations += [cosine_proximity(y_true, y_pred, **params)]
    elif metric == 'log_lik' or metric == 'log_likelihood':
      # Monte Carlo estimate the log-density of the posterior predictive.
      tensor = tf.reduce_mean(output_key.log_prob(y_true))
      log_pred = [sess.run(tensor, feed_dict) for _ in range(n_samples)]
      log_pred = tf.add_n(log_pred) / tf.cast(n_samples, tensor.dtype)
      evaluations += [log_pred]
    elif callable(metric):
      evaluations += [metric(y_true, y_pred, **params)]
    else:
      raise NotImplementedError("Metric is not implemented: {}".format(metric))

  if len(evaluations) == 1:
    return sess.run(evaluations[0], feed_dict)
  else:
    return sess.run(evaluations, feed_dict)


# Classification metrics


def binary_accuracy(y_true, y_pred):
  """Binary prediction accuracy, also known as 0/1-loss.

  Args:
    y_true: tf.Tensor.
      Tensor of 0s and 1s (most generally, any real values a and b).
    y_pred: tf.Tensor.
      Tensor of predictions, with same shape as `y_true`.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def categorical_accuracy(y_true, y_pred):
  """Multi-class prediction accuracy. One-hot representation for `y_true`.

  Args:
    y_true: tf.Tensor.
      Tensor of 0s and 1s, where the outermost dimension of size `K`
      has only one 1 per row.
    y_pred: tf.Tensor.
      Tensor of predictions, with shape `y_true.shape[:-1]`. Each
      entry is an integer {0, 1, ..., K-1}.
  """
  y_true = tf.cast(tf.argmax(y_true, len(y_true.shape) - 1), tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def sparse_categorical_accuracy(y_true, y_pred):
  """Multi-class prediction accuracy. Label {0, 1, .., K-1}
  representation for `y_true`.

  Args:
    y_true: tf.Tensor.
      Tensor of integers {0, 1, ..., K-1}.
    y_pred: tf.Tensor.
      Tensor of predictions, with same shape as `y_true`.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


# Classification metrics (with real-valued predictions)


def binary_crossentropy(y_true, y_pred):
  """Binary cross-entropy.

  Args:
    y_true: tf.Tensor.
      Tensor of 0s and 1s.
    y_pred: tf.Tensor.
      Tensor of real values (logit probabilities), with same shape as
      `y_true`.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def categorical_crossentropy(y_true, y_pred):
  """Multi-class cross entropy. One-hot representation for `y_true`.

  Args:
    y_true: tf.Tensor.
      Tensor of 0s and 1s, where the outermost dimension of size K
      has only one 1 per row.
    y_pred: tf.Tensor.
      Tensor of real values (logit probabilities), with same shape as
      `y_true`. The outermost dimension is the number of classes.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def sparse_categorical_crossentropy(y_true, y_pred):
  """Multi-class cross entropy. Label {0, 1, .., K-1} representation
  for `y_true.`

  Args:
    y_true: tf.Tensor.
      Tensor of integers {0, 1, ..., K-1}.
    y_pred: tf.Tensor.
      Tensor of real values (logit probabilities), with shape
      `(y_true.shape, K)`. The outermost dimension is the number of classes.
  """
  y_true = tf.cast(y_true, tf.int64)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=y_pred, labels=y_true))


def multinomial_accuracy(y_true, y_pred):
  """Multinomial prediction accuracy. `y_true` is a tensor
  of integers, where the outermost dimension gives a draw
  from a Multinomial distribution.

  NB: In evaluating the accuracy between two Multinomials
  results may vary across evaluations. This is because Edward's
  algorithm for computing `y_pred`, i.e. the Multinomial
  mode, yields variable results if `any(isinstance(p, float)
  for p in total_count * probs)` (where `probs` is a vector
  of the predicted Multinomial probabilities).
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def kl_divergence(y_true, y_pred):
  """Kullback-Leibler divergence between two probability distributions. A
  vector of probabilities for `y_true`.

  Args:
    y_true: tf.Tensor.
      Tensor of real values (probabilities) where the values in each row
      of the outermost dimension sum to 1.
    y_pred: tf.Tensor.
      Same as `y_true`, and with the same shape.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  zeros = tf.zeros(shape=(tf.shape(y_true)))
  summand = tf.where(tf.equal(y_true, 0.0), zeros,
                     y_true * (tf.log(y_true) - tf.log(y_pred)))
  return tf.reduce_sum(summand)


def hinge(y_true, y_pred):
  """Hinge loss.

  Args:
    y_true: tf.Tensor.
      Tensor of 0s and 1s.
    y_pred: tf.Tensor.
      Tensor of real values, with same shape as `y_true`.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.maximum(1.0 - y_true * y_pred, 0.0))


def squared_hinge(y_true, y_pred):
  """Squared hinge loss.

  Args:
    y_true: tf.Tensor.
      Tensor of 0s and 1s.
    y_pred: tf.Tensor.
      Tensor of real values, with same shape as `y_true`.
  """
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.square(tf.maximum(1.0 - y_true * y_pred, 0.0)))


# Regression metrics


@with_binary_averaging
def mean_squared_error(y_true, y_pred):
  """Mean squared error loss.

  Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
      Tensors of same shape and type.
  """
  return tf.reduce_mean(tf.square(y_pred - y_true), axis=-2)


@with_binary_averaging
def mean_absolute_error(y_true, y_pred):
  """Mean absolute error loss.

  Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
      Tensors of same shape and type.
  """
  return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-2)


@with_binary_averaging
def mean_absolute_percentage_error(y_true, y_pred):
  """Mean absolute percentage error loss.

  Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
      Tensors of same shape and type.
  """
  diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true),
                                                     1e-8, np.inf))
  return 100.0 * tf.reduce_mean(diff, axis=-2)


@with_binary_averaging
def mean_squared_logarithmic_error(y_true, y_pred):
  """Mean squared logarithmic error loss.

  Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
      Tensors of same shape and type.
  """
  first_log = tf.log(tf.clip_by_value(y_pred, 1e-8, np.inf) + 1.0)
  second_log = tf.log(tf.clip_by_value(y_true, 1e-8, np.inf) + 1.0)
  return tf.reduce_mean(tf.square(first_log - second_log), axis=-2)


def poisson(y_true, y_pred):
  """Negative Poisson log-likelihood of data `y_true` given predictions
  `y_pred` (up to proportion).

  Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
      Tensors of same shape and type.
  """
  return tf.reduce_sum(y_pred - y_true * tf.log(y_pred + 1e-8))


def cosine_proximity(y_true, y_pred):
  """Cosine similarity of two vectors.

  Args:
    y_true: tf.Tensor.
    y_pred: tf.Tensor.
      Tensors of same shape and type.
  """
  y_true = tf.nn.l2_normalize(y_true, len(y_true.shape) - 1)
  y_pred = tf.nn.l2_normalize(y_pred, len(y_pred.shape) - 1)
  return tf.reduce_sum(y_true * y_pred)
