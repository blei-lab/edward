from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.util import logit, get_dims, get_session


def evaluate(metrics, model, variational, data, y_true=None, n_samples=100):
    """Evaluate fitted model using a set of metrics.

    Parameters
    ----------
    metrics : list or str
        List of metrics or a single metric.
    model : ed.Model
        Probability model p(x, z)
    variational : ed.Variational
        Variational approximation to the posterior p(z | x)
    data : dict
        Data dictionary to evaluate model with. For TensorFlow,
        Python, and Stan models, the key type is a string; for PyMC3,
        the key type is a Theano shared variable. For TensorFlow,
        Python, and PyMC3 models, the value type is a NumPy array or
        TensorFlow placeholder; for Stan, the value type is the type
        according to the Stan program's data block.
    y_true : np.ndarray or tf.Tensor
        True values to compare to in supervised learning tasks.
    n_samples : int, optional
        Number of posterior samples for making predictions,
        using the posterior predictive distribution.

    Returns
    -------
    list or float
        A list of evaluations or a single evaluation.

    Raises
    ------
    NotImplementedError
        If an input metric does not match an implemented metric in Edward.
    """
    sess = get_session()
    # Monte Carlo estimate the mean of the posterior predictive:
    # 1. Sample a batch of latent variables from posterior
    zs = variational.sample(n_samples)
    # 2. Make predictions, averaging over each sample of latent variables
    y_pred = model.predict(data, zs)

    # Evaluate y_pred according to y_true for all metrics.
    evaluations = []
    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == 'accuracy' or metric == 'crossentropy':
            # automate binary or sparse cat depending on max(y_true)
            support = tf.reduce_max(y_true).eval()
            if support <= 1:
                metric = 'binary_' + metric
            else:
                metric = 'sparse_categorical_' + metric

        if metric == 'binary_accuracy':
            evaluations += [sess.run(binary_accuracy(y_true, y_pred))]
        elif metric == 'categorical_accuracy':
            evaluations += [sess.run(categorical_accuracy(y_true, y_pred))]
        elif metric == 'sparse_categorical_accuracy':
            evaluations += [sess.run(sparse_categorical_accuracy(y_true, y_pred))]
        elif metric == 'log_loss' or metric == 'binary_crossentropy':
            evaluations += [sess.run(binary_crossentropy(y_true, y_pred))]
        elif metric == 'categorical_crossentropy':
            evaluations += [sess.run(categorical_crossentropy(y_true, y_pred))]
        elif metric == 'sparse_categorical_crossentropy':
            evaluations += [sess.run(sparse_categorical_crossentropy(y_true, y_pred))]
        elif metric == 'hinge':
            evaluations += [sess.run(hinge(y_true, y_pred))]
        elif metric == 'squared_hinge':
            evaluations += [sess.run(squared_hinge(y_true, y_pred))]
        elif metric == 'mse' or metric == 'MSE' or \
             metric == 'mean_squared_error':
            evaluations += [sess.run(mean_squared_error(y_true, y_pred))]
        elif metric == 'mae' or metric == 'MAE' or \
             metric == 'mean_absolute_error':
            evaluations += [sess.run(mean_absolute_error(y_true, y_pred))]
        elif metric == 'mape' or metric == 'MAPE' or \
             metric == 'mean_absolute_percentage_error':
            evaluations += [sess.run(mean_absolute_percentage_error(y_true, y_pred))]
        elif metric == 'msle' or metric == 'MSLE' or \
             metric == 'mean_squared_logarithmic_error':
            evaluations += [sess.run(mean_squared_logarithmic_error(y_true, y_pred))]
        elif metric == 'poisson':
            evaluations += [sess.run(poisson(y_true, y_pred))]
        elif metric == 'cosine' or metric == 'cosine_proximity':
            evaluations += [sess.run(cosine_proximity(y_true, y_pred))]
        elif metric == 'log_lik' or metric == 'log_likelihood':
            evaluations += [sess.run(y_pred)]
        else:
            raise NotImplementedError()

    if len(evaluations) == 1:
        return evaluations[0]
    else:
        return evaluations


def ppc(model, variational=None, data=None, T=None, n_samples=100):
    """Posterior predictive check.
    (Rubin, 1984; Meng, 1994; Gelman, Meng, and Stern, 1996)
    If no posterior approximation is provided through ``variational``,
    then we default to a prior predictive check (Box, 1980).

    PPC's form an empirical distribution for the predictive discrepancy,

    .. math::
        p(T) = \int p(T(xrep) | z) p(z | x) dz

    by drawing replicated data sets xrep and calculating
    :math:`T(xrep)` for each data set. Then it compares it to
    :math:`T(x)`.

    Parameters
    ----------
    model : ed.Model
        Class object that implements the ``sample_likelihood`` method
    variational : ed.Variational, optional
        Latent variable distribution q(z) to sample from. It is an
        approximation to the posterior, e.g., a variational
        approximation or an empirical distribution from MCMC samples.
        If not specified, samples will be obtained from the model
        through the ``sample_prior`` method.
    data : dict, optional
        Observed data to compare to. If not specified, will return
        only the reference distribution with an assumed replicated
        data set size of 1. For TensorFlow, Python, and Stan models,
        the key type is a string; for PyMC3, the key type is a Theano
        shared variable. For TensorFlow, Python, and PyMC3 models, the
        value type is a NumPy array or TensorFlow placeholder; for
        Stan, the value type is the type according to the Stan
        program's data block.
    T : function, optional
        Discrepancy function, which takes a data dictionary and list
        of latent variables as input and outputs a tf.Tensor. Default
        is the identity function.
    n_samples : int, optional
        Number of replicated data sets.

    Returns
    -------
    list
        List containing the reference distribution, which is a Numpy
        vector of size elements,

        .. math::
            (T(xrep^{1}, z^{1}), ..., T(xrep^{size}, z^{size}))

        and the realized discrepancy, which is a NumPy vector of size
        elements,

        .. math::
            (T(x, z^{1}), ..., T(x, z^{size})).

        If the discrepancy function is not specified, then the list
        contains the full data distribution where each element is a
        data set (dictionary).
    """
    sess = get_session()
    if data is None:
        N = 1
        x = {}
    else:
        # Assume all values have the same data set size.
        N = get_dims(list(six.itervalues(data))[0])[0]
        x = data

    # 1. Sample from posterior (or prior).
    # We must fetch zs out of the session because sample_likelihood()
    # may require a SciPy-based sampler.
    if variational is not None:
        zs = variational.sample(n_samples)
        # This is to avoid fetching, e.g., a placeholder x with the
        # dictionary {x: np.array()}. TensorFlow will raise an error.
        if isinstance(zs, list):
            zs = [tf.identity(zs_elem) for zs_elem in zs]
        else:
            zs = tf.identity(zs)

        zs = sess.run(zs)
    else:
        zs = model.sample_prior(n_samples)
        zs = zs.eval()

    # 2. Sample from likelihood.
    xreps = model.sample_likelihood(zs, N)

    # 3. Calculate discrepancy.
    if T is None:
        if x is None:
            return xreps
        else:
            return [xreps, y]

    Txreps = []
    Txs = []
    for xrep, z in zip(yreps, tf.unpack(zs)):
        Txreps += [T(xrep, z)]
        if y is not None:
            Txs += [T(x, z)]

    if x is None:
        return sess.run(tf.pack(Txreps))
    else:
        return sess.run([tf.pack(Txreps), tf.pack(Txs)])


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
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))


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
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true))


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
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, y_true))


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
    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, np.inf))
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
