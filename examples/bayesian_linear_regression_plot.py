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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
        self.n_vars = 2

    def log_prob(self, xs, zs):
        """Return a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        x, y = xs['x'], xs['y']
        log_prior = -tf.reduce_sum(zs*zs, 1) / self.prior_variance
        # broadcasting to do (x*W) + b (n_minibatch x n_samples - n_samples)
        W = tf.expand_dims(zs[:, 0], 0)
        b = zs[:, 1]
        mus = tf.matmul(x, W) + b
        # broadcasting to do mus - y (n_minibatch x n_samples - n_minibatch x 1)
        y = tf.expand_dims(y, 1)
        log_lik = -tf.reduce_sum(tf.pow(mus - y, 2), 0) / self.lik_variance
        return log_lik + log_prior


def build_toy_dataset(N=40, noise_std=0.1):
    ed.set_seed(0)
    x  = np.concatenate([np.linspace(0, 2, num=N/2),
                         np.linspace(6, 8, num=N/2)])
    y = 0.075*x + norm.rvs(0, noise_std, size=N)
    x = (x - 4.0) / 4.0
    x = x.reshape((N, 1))
    return {'x': x, 'y': y}


ed.set_seed(42)
model = LinearModel()
variational = Variational()
variational.add(Normal(model.n_vars))
data = build_toy_dataset()

# Set up figure
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

sess = ed.get_session()
inference = ed.MFVI(model, variational, data)
inference.initialize(n_samples=5, n_print=5)
for t in range(250):
    loss = inference.update()
    if t % inference.n_print == 0:
        print("iter {:d} loss {:.2f}".format(t, loss))

        # Sample functions from variational model
        mean, std = sess.run([variational.layers[0].loc,
                              variational.layers[0].scale])
        rs = np.random.RandomState(0)
        zs = rs.randn(10, variational.n_vars) * std + mean
        zs = tf.constant(zs, dtype=tf.float32)
        inputs = np.linspace(-8, 8, num=400, dtype=np.float32)
        x = tf.expand_dims(tf.constant(inputs), 1)
        W = tf.expand_dims(zs[:, 0], 0)
        b = zs[:, 1]
        mus = tf.matmul(x, W) + b
        outputs = mus.eval()

        # Get data
        x, y = data['x'], data['y']

        # Plot data and functions
        plt.cla()
        ax.plot(x, y, 'bx')
        ax.plot(inputs, outputs)
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)
