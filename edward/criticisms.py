import numpy as np
import tensorflow as tf

# TODO default to grabbing session from environment if it exists
# to do this, inference will need to globally define the session
def evaluate(metrics, model, variational, data, sess=tf.Session()):
    """
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
    # 2. Form a set of predictions for each sample of latent variables
    y_pred_zs = model.predict(xs, zs)
    # 3. Average over set of predictions
    y_pred = tf.reduce_mean(y_pred_zs, 1)
    # TODO
    y_true = data.data[:, 0]

    # Evaluate y_pred according to y_true for all metrics.
    evaluations = []
    if isinstance(metrics, str):
        metrics = [metrics]

    for metric in metrics:
        # TODO
        # metric == 'accuracy', with binary and categorical defaulted
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
        else:
            raise NotImplementedError()

    if len(evaluations) == 1:
        return evaluations[0]
    else:
        return evaluations

def cv_evaluate(metric, model, variational, data, sess=tf.Session()):
    """
    Cross-validated evaluation
    """
    # TODO it calls evaluate(), wrapped around importance sampling
    raise NotImplementedError()

def ppc(model, variational, data, T, size=100, sess=tf.Session()):
    """
    Posterior predictive check.
    (Rubin, 1984; Meng, 1994; Gelman, Meng, and Stern, 1996)

    It form an empirical distribution for the predictive discrepancy,
    p(T) = \int p(T(x) | z) p(z | x) dz
    by drawing replicated data sets xrep and calculating T(xrep) for
    each data set. Then it compares it to T(xobs).

    Parameters
    ----------
    model : Model
        model object must have a 'sample_lik' method, which takes xs,
        zs, size as input and returns replicated data set
    data : Data
        Observed data to check to.
    variational : Variational
        latent variable distribution q(z) to sample from. It is an
        approximation to the posterior, e.g., a variational
        approximation or an empirical distribution from MCMC samples.
    T : function
        Discrepancy function.
    size : int, optional
        number of replicated data sets
    sess : tf.Session, optional
        session used during inference

    Returns
    -------
    list
        List containing the reference distribution, which is a Numpy
        vector of size elements,
        (T(xrep^{1}, z^{1}), ..., T(xrep^{size}, z^{size}));
        and the realized discrepancy, which is a NumPy array of size
        elements,
        (T(x, z^{1}), ..., T(x, z^{size})).
    """
    # TODO
    xobs = sess.run(data.data) # TODO generalize to arbitrary data
    Txobs = T(xobs)
    N = len(xobs) # TODO len, or shape[0]

    # TODO
    # size in variational sample
    # whether the sample method requires sess
    zreps = latent.sample([size, 1], sess)
    xreps = [model.sample_likelihood(zrep, N) for zrep in zreps]
    Txreps = [T(xrep) for xrep in xreps]
    return Txobs, Txreps

# TODO maybe it should default to prior PC if variational is not given as
# input
def prior_predictive_check(model, data, T):
    """
    Prior predictive check.
    (Box, 1980)

    It form an empirical distribution for the predictive discrepancy,
    p(T) = \int p(T(x) | z) p(z) dz
    by drawing replicated data sets xrep and calculating T(xrep) for
    each data set. Then it compares it to T(xobs).

    Parameters
    ----------
    model : Model
        model object must have a 'sample_lik' method, which takes xs,
        zs, size as input and returns replicated data set.
        model object must have a 'sample_prior' method, which takes xs,
        zs, size as input and returns...
    data : Data
        Observed data to check to.
    T : function
        Discrepancy function.
    size : int, optional
        number of replicated data sets
    sess : tf.Session, optional
        session used during inference

    Returns
    -------
    list
        List containing the reference distribution, which is a Numpy
        vector of size elements,
        (T(xrep^{1}, z^{1}), ..., T(xrep^{size}, z^{size}));
        and the realized discrepancy, which is a NumPy array of size
        elements,
        (T(x, z^{1}), ..., T(x, z^{size})).
    """
    raise NotImplementedError()

# Classification metrics

# TODO maybe i should be doing these rounding stuff outside
# for accuracy, it makes sense to hand it the actual values
# but not for hinge of cross entropies
def binary_accuracy(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor of 0s and 1s.
    y_pred : tf.Tensor
        Tensor of probabilities.
    """
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(y_pred)), tf.float32))

# TODO double check
def categorical_accuracy(y_true, y_pred):
    """
    One-hot representation.
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor of integers {0, 1, ..., K-1}.
    y_pred : tf.Tensor
        Tensor of probabilities, with shape (y_true.get_shape(), K).
        The outermost dimension are the categorical probabilities for
        that data point.
    """
    return tf.reduce_mean(tf.cast(tf.equal(y_true,
           tf.argmax(y_pred, len(y_pred.get_shape()) - 1)), tf.float32))

# TODO double check
def sparse_categorical_accuracy(y_true, y_pred):
    """Label {0, 1, .., K-1} representation."""
    return tf.reduce_mean(tf.cast(tf.equal(
           tf.reduce_max(y_true, len(y_true.get_shape()) - 1),
           tf.cast(tf.argmax(y_pred, len(y_pred.get_shape()) - 1), tf.float32)),
           tf.float32))

# TODO double check
def binary_crossentropy(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))

# TODO double check
def categorical_crossentropy(y_true, y_pred):
    """Expects a binary class matrix instead of a vector of scalar classes.
    """
    return tf.reduce_mean(tf.categorical_crossentropy(y_pred, y_true))

# TODO double check
def sparse_categorical_crossentropy(y_true, y_pred):
    """expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        y_pred, tf.cast(y_true, tf.float32)))

def hinge(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor of 0s and 1s.
    y_pred : tf.Tensor
        Tensor of probabilities.
    """
    return tf.reduce_mean(tf.reduce_max(1.0 - y_true * y_pred, 0.0))

def squared_hinge(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : tf.Tensor
        Tensor of 0s and 1s.
    y_pred : tf.Tensor
        Tensor of probabilities.
    """
    return tf.reduce_mean(tf.square(tf.reduce_max(1.0 - y_true * y_pred, 0.0)))

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
    Negative log Poisson likelihood of data y_true given predictions
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

# Unsupervised metrics
# TODO
# log_likelihood(), log p(y_true) under model's marginal likelihood
