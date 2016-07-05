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
import matplotlib.pyplot as plt
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
        self.num_vars = 2

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        # Data has output in first column and input in second column.
        y = xs[:, 0]
        x = xs[:, 1]
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs, 1)
        # broadcasting to do (x*W) + b (n_data x n_minibatch - n_minibatch)
        x = tf.expand_dims(x, 1)
        W = tf.expand_dims(zs[:, 0], 0)
        b = zs[:, 1]
        mus = tf.matmul(x, W) + b
        # broadcasting to do mus - y (n_data x n_minibatch - n_data x 1)
        y = tf.expand_dims(y, 1)
        log_lik = -tf.reduce_sum(tf.pow(mus - y, 2), 0) / self.lik_variance
        return log_lik + log_prior

def build_toy_dataset(n_data=40, noise_std=0.1):
    ed.set_seed(0)
    x  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                         np.linspace(6, 8, num=n_data/2)])
    y = 0.075*x + norm.rvs(0, noise_std, size=n_data)
    x = (x - 4.0) / 4.0
    x = x.reshape((n_data, 1))
    y = y.reshape((n_data, 1))
    data = np.concatenate((y, x), axis=1) # n_data x 2
    data = tf.constant(data, dtype=tf.float32)
    return ed.Data(data)

ed.set_seed(42)
model = LinearModel()
variational = Variational()
variational.add(Normal(model.num_vars))
data = build_toy_dataset()

# Set up figure
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

sess = ed.get_session()
inference = ed.MFVI(model, variational, data)
inference.initialize(n_minibatch=5, n_print=5)
for t in range(250):
    loss = inference.update()
    if t % inference.n_print == 0:
        print("iter {:d} loss {:.2f}".format(t, loss))

        # Sample functions from variational model
        mean, std = sess.run([variational.layers[0].loc,
                              variational.layers[0].scale])
        rs = np.random.RandomState(0)
        zs = rs.randn(10, variational.num_vars) * std + mean
        zs = tf.constant(zs, dtype=tf.float32)
        inputs = np.linspace(-8, 8, num=400, dtype=np.float32)
        x = tf.expand_dims(tf.constant(inputs), 1)
        W = tf.expand_dims(zs[:, 0], 0)
        b = zs[:, 1]
        mus = tf.matmul(x, W) + b
        outputs = mus.eval()

        # Get data
        y, x = sess.run([data.data[:, 0], data.data[:, 1]])

        # Plot data and functions
        plt.cla()
        ax.plot(x, y, 'bx')
        ax.plot(inputs, outputs)
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)
