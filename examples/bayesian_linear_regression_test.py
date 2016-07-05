#!/usr/bin/env python
"""
Bayesian linear regression using mean-field variational inference.

Probability model:
    Bayesian linear model
    Prior: Normal
    Likelihood: Normal
Variational model
    Likelihood: Mean-field Normal
"""
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational, Normal
from edward.stats import norm

class LinearModel:
    """
    Bayesian linear regression for outputs y on inputs x.

    p((x,y), z) = Normal(y | x*z, lik_variance) *
                  Normal(z | 0, prior_variance),

    where z are weights, and with known lik_variance and
    prior_variance.

    Parameters
    ----------
    lik_variance : float, optional
        Variance of the normal likelihood; aka noise parameter,
        homoscedastic variance, scale parameter.
    prior_variance : float, optional
        Variance of the normal prior on weights; aka L2
        regularization parameter, ridge penalty, scale parameter.
    """
    def __init__(self, lik_variance=0.01, prior_variance=0.01):
        self.lik_variance = lik_variance
        self.prior_variance = prior_variance
        self.num_vars = 11

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        # Data has output in first column and input in remaining columns.
        x, y = xs['x'], xs['y']
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs, 1)
        b = zs[:, 0]
        W = tf.transpose(zs[:, 1:])
        mus = tf.matmul(x, W) + b
        y = tf.expand_dims(y, 1)
        log_lik = -tf.reduce_sum(tf.pow(mus - y, 2), 0) / self.lik_variance
        return log_lik + log_prior

    def predict(self, xs, zs):
        """Returns a prediction for each data point, averaging over
        each set of latent variables z in zs; and also return the true
        value."""
        x_test = xs['x']
        b = zs[:, 0]
        W = tf.transpose(zs[:, 1:])
        y_pred = tf.reduce_mean(tf.matmul(x_test, W) + b, 1)
        return y_pred

def build_toy_dataset(n_data=40, coeff=np.random.randn(10), noise_std=0.1):
    n_dim = len(coeff)
    x = np.random.randn(n_data, n_dim)
    y = np.dot(x, coeff) + norm.rvs(0, noise_std, size=n_data).reshape((n_data,))
    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y.reshape((n_data, 1)), dtype=tf.float32)
    return {'x': x, 'y': y}

ed.set_seed(42)
model = LinearModel()
variational = Variational()
variational.add(Normal(model.num_vars))

data = build_toy_dataset()

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=250, n_minibatch=5, n_print=10)

data_test = build_toy_dataset()
x_test, y_test = data_test['x'], data_test['y']
print(ed.evaluate('mse', model, variational, {'x': x_test}, y_test))
