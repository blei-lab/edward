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
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, weight_dim, inv_link=tf.sigmoid, prior_variance=0.01):
        self.weight_dim = weight_dim
        self.inv_link = inv_link
        self.prior_variance = prior_variance
        self.num_vars = (self.weight_dim[0]+1)*self.weight_dim[1]

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
        h = tf.squeeze(h) # n_data x 1 to n_data
        return h

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        # Data must have labels in the first column and features in
        # subsequent columns.
        y = xs[:, 0]
        x = xs[:, 1:]
        log_lik = []
        for z in tf.unpack(zs):
            p = self.mapping(x, z)
            log_lik += [bernoulli.logpmf(y, p)]

        log_lik = tf.pack(log_lik)
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs, 1)
        return log_lik + log_prior

def build_toy_dataset(n_data=40, noise_std=0.1):
    ed.set_seed(0)
    D = 1
    x  = np.linspace(-3, 3, num=n_data)
    y = np.tanh(x) + norm.rvs(0, noise_std, size=n_data)
    y[y < 0.5] = 0
    y[y >= 0.5] = 1
    x = (x - 4.0) / 4.0
    x = x.reshape((n_data, D))
    y = y.reshape((n_data, 1))
    data = np.concatenate((y, x), axis=1) # n_data x (D+1)
    data = tf.constant(data, dtype=tf.float32)
    return ed.Data(data)

ed.set_seed(42)
model = HierarchicalLogistic(weight_dim=[1,1])
variational = Variational()
variational.add(Normal(model.num_vars))
data = build_toy_dataset()

# Set up figure
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

def print_progress(self, t, losses, sess):
    if t % self.n_print == 0:
        print("iter %d loss %.2f " % (t, np.mean(losses)))
        self.variational.print_params(sess)

        # Sample functions from variational model
        mean, std = sess.run([self.variational.layers[0].m,
                              self.variational.layers[0].s])
        rs = np.random.RandomState(0)
        zs = rs.randn(10, self.variational.num_vars) * std + mean
        zs = tf.constant(zs, dtype=tf.float32)
        inputs = np.linspace(-3, 3, num=400, dtype=np.float32)
        x = tf.expand_dims(tf.constant(inputs), 1)
        mus = tf.pack([self.model.mapping(x, z) for z in tf.unpack(zs)])
        outputs = sess.run(mus)

        # Get data
        y, x = sess.run([self.data.data[:, 0], self.data.data[:, 1]])

        # Plot data and functions
        plt.cla()
        ax.plot(x, y, 'bx')
        ax.plot(inputs, outputs.T)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-0.5, 1.5])
        plt.draw()

ed.MFVI.print_progress = print_progress
inference = ed.MFVI(model, variational, data)
# TODO it gets NaN's at iteration 608 and beyond
inference.run(n_iter=600, n_print=5)
