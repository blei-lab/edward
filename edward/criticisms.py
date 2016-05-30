import numpy as np
import tensorflow as tf

from edward.data import Data
from edward.util import logit

def evaluate(metrics, model, variational, data, sess=tf.Session()):
    """
    Evaluate fitted model using a set of metrics.

    Parameters
    ----------
    metric : list or str
        List of metrics or a single metric.

    Returns
    -------
    list or float
        A list of evaluations or a single evaluation.
    """
    # Monte Carlo estimate the mean of the posterior predictive:
    # 1. Sample a batch of latent variables from posterior
    xs = data.data
    n_minibatch = 100
    zs, samples = variational.sample(xs, size=n_minibatch)
    feed_dict = variational.np_sample(samples, n_minibatch, sess=sess)
    # 2. Make predictions, averaging over each sample of latent variables
    y_pred, y_true = model.predict(xs, zs)

    # Evaluate y_pred according to y_true for all metrics.
    evaluations = []
    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        if metric == 'accuracy' or metric == 'crossentropy':
            # automate binary or sparse cat depending on max(y_true)
            support = sess.run(tf.reduce_max(y_true))
            if support <= 1:
                metric = 'binary_' + metric
            else:
                metric = 'sparse_categorical_' + metric

        if metric == 'binary_accuracy':
            evaluations += [sess.run(binary_accuracy(y_true, y_pred), feed_dict)]
        elif metric == 'categorical_accuracy':
            evaluations += [sess.run(categorical_accuracy(y_true, y_pred), feed_dict)]
        elif metric == 'sparse_categorical_accuracy':
            evaluations += [sess.run(sparse_categorical_accuracy(y_true, y_pred), feed_dict)]
        elif metric == 'log_loss' or metric == 'binary_crossentropy':
            evaluations += [sess.run(binary_crossentropy(y_true, y_pred), feed_dict)]
        elif metric == 'categorical_crossentropy':
            evaluations += [sess.run(categorical_crossentropy(y_true, y_pred), feed_dict)]
        elif metric == 'sparse_categorical_crossentropy':
            evaluations += [sess.run(sparse_categorical_crossentropy(y_true, y_pred), feed_dict)]
        elif metric == 'hinge':
            evaluations += [sess.run(hinge(y_true, y_pred), feed_dict)]
        elif metric == 'squared_hinge':
            evaluations += [sess.run(squared_hinge(y_true, y_pred), feed_dict)]
        elif metric == 'mse' or metric == 'MSE' or \
             metric == 'mean_squared_error':
            evaluations += [sess.run(mean_squared_error(y_true, y_pred), feed_dict)]
        elif metric == 'mae' or metric == 'MAE' or \
             metric == 'mean_absolute_error':
            evaluations += [sess.run(mean_absolute_error(y_true, y_pred), feed_dict)]
        elif metric == 'mape' or metric == 'MAPE' or \
             metric == 'mean_absolute_percentage_error':
            evaluations += [sess.run(mean_absolute_percentage_error(y_true, y_pred), feed_dict)]
        elif metric == 'msle' or metric == 'MSLE' or \
             metric == 'mean_squared_logarithmic_error':
            evaluations += [sess.run(mean_squared_logarithmic_error(y_true, y_pred), feed_dict)]
        elif metric == 'poisson':
            evaluations += [sess.run(poisson(y_true, y_pred), feed_dict)]
        elif metric == 'cosine' or metric == 'cosine_proximity':
            evaluations += [sess.run(cosine_proximity(y_true, y_pred), feed_dict)]
        elif metric == 'log_lik' or metric == 'log_likelihood':
            evaluations += [sess.run(y_pred, feed_dict)]
        else:
            raise NotImplementedError()

    if len(evaluations) == 1:
        return evaluations[0]
    else:
        return evaluations

def ppc(model, variational=None, data=Data(), T=None, size=100,
    sess=tf.Session()):
    """
    Posterior predictive check.
    (Rubin, 1984; Meng, 1994; Gelman, Meng, and Stern, 1996)
    If variational is not specified, it defaults to a prior predictive
    check (Box, 1980).

    PPC's form an empirical distribution for the predictive discrepancy,
    p(T) = \int p(T(yrep) | z) p(z | y) dz
    by drawing replicated data sets yrep and calculating T(yrep) for
    each data set. Then it compares it to T(y).

    Parameters
    ----------
    model : Model
        class object with a 'sample_likelihood' method
    variational : Variational, optional
        latent variable distribution q(z) to sample from. It is an
        approximation to the posterior, e.g., a variational
        approximation or an empirical distribution from MCMC samples.
        If not specified, samples will be obtained from model
        with a 'sample_prior' method.
    data : Data, optional
        Observed data to compare to. If not specified, will return
        only the reference distribution with an assumed replicated
        data set size of 1.
    T : function, optional
        Discrepancy function written in TensorFlow. Default is
        identity. It is a function taking in a data set
        y and optionally a set of latent variables z as input.
    size : int, optional
        number of replicated data sets
    sess : tf.Session, optional
        session used during inference

    Returns
    -------
    list
        List containing the reference distribution, which is a Numpy
        vector of size elements,
        (T(yrep^{1}, z^{1}), ..., T(yrep^{size}, z^{size}));
        and the realized discrepancy, which is a NumPy vector of size
        elements,
        (T(y, z^{1}), ..., T(y, z^{size})).
    """
    y = data.data
    if y == None:
        N = 1
    else:
        N = data.N

    if T == None:
        T = lambda y, z=None: y

    # 1. Sample from posterior (or prior).
    # We must fetch zs out of the session because sample_likelihood()
    # may require a SciPy-based sampler.
    if variational != None:
        zs, samples = variational.sample(y, size=size)
        feed_dict = variational.np_sample(samples, size, sess=sess)
        zs = sess.run(zs, feed_dict)
    else:
        zs = model.sample_prior(size=size)
        zs = sess.run(zs)

    # 2. Sample from likelihood.
    yreps = model.sample_likelihood(zs, size=N)
    # 3. Calculate discrepancy.
    Tyreps = []
    Tys = []
    for yrep, z in zip(yreps, tf.unpack(zs)):
        Tyreps += [T(yrep, z)]
        if y != None:
            Tys += [T(y, z)]

    if y == None:
        return sess.run(tf.pack(Tyreps), feed_dict)
    else:
        return sess.run([tf.pack(Tyreps), tf.pack(Tys)], feed_dict)

# Classification metrics

def binary_accuracy(y_true, y_pred):
    """
    Binary prediction accuracy, also known as 0/1-loss.

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
    """
    Multi-class prediction accuracy. One-hot representation for
    y_true.

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
    y_true = tf.cast(tf.argmax(y_true, len(y_true.get_shape()) - 1), tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, len(y_pred.get_shape()) - 1), tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

def sparse_categorical_accuracy(y_true, y_pred):
    """
    Multi-class prediction accuracy. Label {0, 1, .., K-1}
    representation for y_true.

    Parameters
    ----------
    y_true : tf.Tensor
        Tensor of integers {0, 1, ..., K-1}.
    y_pred : tf.Tensor
        Tensor of probabilities, with shape (y_true.get_shape(), K).
        The outermost dimension are the categorical probabilities for
        that data point.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, len(y_pred.get_shape()) - 1), tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

def binary_crossentropy(y_true, y_pred):
    """
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
    """
    Multi-class cross entropy. One-hot representation for y_true.

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
    """
    Multi-class cross entropy. Label {0, 1, .., K-1} representation
    for y_true.

    Parameters
    ----------
    y_true : tf.Tensor
        Tensor of integers {0, 1, ..., K-1}.
    y_pred : tf.Tensor
        Tensor of probabilities, with shape (y_true.get_shape(), K).
        The outermost dimension are the categorical probabilities for
        that data point.
    """
    y_true = tf.cast(y_true, tf.int64)
    y_pred = logit(tf.cast(y_pred, tf.float32))
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, y_true))

def hinge(y_true, y_pred):
    """
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
    """
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
    """
    Parameters
    ----------
    y_true : tf.Tensor
    y_pred : tf.Tensor
        Tensors of same shape and type.
    """
    return tf.reduce_mean(tf.square(y_pred - y_true))

def mean_absolute_error(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : tf.Tensor
    y_pred : tf.Tensor
        Tensors of same shape and type.
    """
    return tf.reduce_mean(tf.abs(y_pred - y_true))

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : tf.Tensor
    y_pred : tf.Tensor
        Tensors of same shape and type.
    """
    diff = tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, np.inf))
    return 100.0 * tf.reduce_mean(diff)

def mean_squared_logarithmic_error(y_true, y_pred):
    """
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
    """
    Negative Poisson log-likelihood of data y_true given predictions
    y_pred (up to proportion).

    Parameters
    ----------
    y_true : tf.Tensor
    y_pred : tf.Tensor
        Tensors of same shape and type.
    """
    return tf.reduce_sum(y_pred - y_true * tf.log(y_pred + 1e-8))

def cosine_proximity(y_true, y_pred):
    """
    Cosine similarity of two vectors.

    Parameters
    ----------
    y_true : tf.Tensor
    y_pred : tf.Tensor
        Tensors of same shape and type.
    """
    y_true = tf.nn.l2_normalize(y_true, len(y_true.get_shape()) - 1)
    y_pred = tf.nn.l2_normalize(y_pred, len(y_pred.get_shape()) - 1)
    return tf.reduce_sum(y_true * y_pred)
