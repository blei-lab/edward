#!/usr/bin/env python
"""
Bayesian neural network using mean-field variational inference.
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016))
Inspired by autograd's Bayesian neural network example.

Probability model:
    Bayesian neural network
    Prior: Normal
    Likelihood: Normal with mean parameterized by fully connected NN
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
from edward.util import rbf


class BayesianNN:
    """
    Bayesian neural network for regressing outputs y on inputs x.

    p((x,y), z) = Normal(y | NN(x; z), lik_variance) *
                  Normal(z | 0, prior_variance),

    where z are neural network weights, and with known lik_variance
    and prior_variance.

    Parameters
    ----------
    layer_sizes : list
        The size of each layer, ordered from input to output.
    nonlinearity : function, optional
        Non-linearity after each linear transformation in the neural
        network; aka activation function.
    lik_variance : float, optional
        Variance of the normal likelihood; aka noise parameter,
        homoscedastic variance, scale parameter.
    prior_variance : float, optional
        Variance of the normal prior on weights; aka L2
        regularization parameter, ridge penalty, scale parameter.
    """
    def __init__(self, layer_sizes, nonlinearity=tf.nn.tanh,
        lik_variance=0.01, prior_variance=1):
        self.layer_sizes = layer_sizes
        self.nonlinearity = nonlinearity
        self.lik_variance = lik_variance
        self.prior_variance = prior_variance

        self.n_layers = len(layer_sizes)
        self.weight_dims = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.n_vars = sum((m+1)*n for m, n in self.weight_dims)

    def unpack_weights(self, z):
        """Unpack weight matrices and biases from a flattened vector."""
        for m, n in self.weight_dims:
            yield tf.reshape(z[:m*n],        [m, n]), \
                  tf.reshape(z[m*n:(m*n+n)], [1, n])
            z = z[(m+1)*n:]

    def neural_network(self, x, zs):
        """
        Return a `n_samples` x `n_minibatch` matrix. Each row is
        the output of a neural network on the input data `x` and
        given a set of weights `z` in `zs`.
        """
        matrix = []
        for z in tf.unpack(zs):
            # Calculate neural network with weights given by `z`.
            h = x
            for W, b in self.unpack_weights(z):
                # broadcasting to do (h*W) + b (e.g. 40x10 + 1x10)
                h = self.nonlinearity(tf.matmul(h, W) + b)

            matrix += [tf.squeeze(h)] # n_minibatch x 1 to n_minibatch

        return tf.pack(matrix)

    def log_prob(self, xs, zs):
        """Return a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        x, y = xs['x'], xs['y']
        log_prior = -tf.reduce_sum(zs*zs, 1) / self.prior_variance
        mus = self.neural_network(x, zs)
        # broadcasting to do mus - y (n_samples x n_minibatch - n_minibatch)
        log_lik = -tf.reduce_sum(tf.pow(mus - y, 2), 1) / self.lik_variance
        return log_lik + log_prior


def build_toy_dataset(N=40, noise_std=0.1):
    ed.set_seed(0)
    D = 1
    x  = np.concatenate([np.linspace(0, 2, num=N/2),
                         np.linspace(6, 8, num=N/2)])
    y = np.cos(x) + norm.rvs(0, noise_std, size=N)
    x = (x - 4.0) / 4.0
    x = x.reshape((N, D))
    return {'x': x, 'y': y}


ed.set_seed(42)
model = BayesianNN(layer_sizes=[1, 10, 10, 1], nonlinearity=rbf)
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
inference.initialize(n_print=100)
for t in range(1000):
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
        mus = model.neural_network(x, zs)
        outputs = mus.eval()

        # Get data
        x, y = data['x'], data['y']

        # Plot data and functions
        plt.cla()
        ax.plot(x, y, 'bx')
        ax.plot(inputs, outputs.T)
        ax.set_xlim([-8, 8])
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)
