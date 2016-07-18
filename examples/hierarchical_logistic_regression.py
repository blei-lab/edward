#!/usr/bin/env python
"""
Hierarchical logistic regression using mean-field variational inference.

Probability model:
    Hierarchical logistic regression
    Prior: Normal
    Likelihood: Bernoulli-Logit
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
from edward.stats import bernoulli, norm


class HierarchicalLogistic:
    """
    Hierarchical logistic regression for outputs y on inputs x.

    p((x,y), z) = Bernoulli(y | link^{-1}(x*z)) *
                  Normal(z | 0, prior_variance),

    where z are weights, and with known link function and
    prior_variance.

    Parameters
    ----------
    weight_dim : list
        Dimension of weights, which is dimension of input x dimension
        of output.
    inv_link : function, optional
        Inverse of link function, which is applied to the linear transformation.
    prior_variance : float, optional
        Variance of the normal prior on weights; aka L2
        regularization parameter, ridge penalty, scale parameter.
    """
    def __init__(self, weight_dim, inv_link=tf.sigmoid, prior_variance=10):
        self.weight_dim = weight_dim
        self.inv_link = inv_link
        self.prior_variance = prior_variance
        self.n_vars = (self.weight_dim[0]+1)*self.weight_dim[1]

    def mapping(self, x, z):
        """
        Inverse link function on linear transformation,
        link^{-1}(W*x + b)
        """
        m, n = self.weight_dim[0], self.weight_dim[1]
        W = tf.reshape(z[:m*n], [m, n])
        b = tf.reshape(z[m*n:], [1, n])
        # broadcasting to do (x*W) + b (e.g. 40x10 + 1x10)
        h = self.inv_link(tf.matmul(x, W) + b)
        h = tf.squeeze(h) # n_minibatch x 1 to n_minibatch
        return h

    def log_prob(self, xs, zs):
        """Return a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        x, y = xs['x'], xs['y']
        log_lik = []
        for z in tf.unpack(zs):
            p = self.mapping(x, z)
            log_lik += [bernoulli.logpmf(y, p)]

        log_lik = tf.pack(log_lik)
        log_prior = -tf.reduce_sum(zs*zs, 1) / self.prior_variance
        return log_lik + log_prior


def build_toy_dataset(N=40, noise_std=0.1):
    ed.set_seed(0)
    D = 1
    x  = np.linspace(-3, 3, num=N)
    y = np.tanh(x) + norm.rvs(0, noise_std, size=N)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    x = (x - 4.0) / 4.0
    x = x.reshape((N, D))
    return {'x': x, 'y': y}


ed.set_seed(42)
model = HierarchicalLogistic(weight_dim=[1,1])
variational = Variational()
variational.add(Normal(model.n_vars))
data = build_toy_dataset()

# Set up figure
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

inference = ed.MFVI(model, variational, data)
inference.initialize(n_print=5)
sess = ed.get_session()
for t in range(600):
    loss = inference.update()
    if t % inference.n_print == 0:
        print("iter {:d} loss {:.2f}".format(t, loss))
        print(variational)

        # Sample functions from variational model
        mean, std = sess.run([variational.layers[0].loc,
                              variational.layers[0].scale])
        rs = np.random.RandomState(0)
        zs = rs.randn(10, variational.n_vars) * std + mean
        zs = tf.constant(zs, dtype=tf.float32)
        inputs = np.linspace(-3, 3, num=400, dtype=np.float32)
        x = tf.expand_dims(tf.constant(inputs), 1)
        mus = tf.pack([model.mapping(x, z) for z in tf.unpack(zs)])
        outputs = mus.eval()

        # Get data
        x, y = data['x'], data['y']

        # Plot data and functions
        plt.cla()
        ax.plot(x, y, 'bx')
        ax.plot(inputs, outputs.T)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-0.5, 1.5])
        plt.draw()
        plt.pause(1.0/60.0)
