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
import edward as ed
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

        self.num_layers = len(layer_sizes)
        self.weight_dims = zip(layer_sizes[:-1], layer_sizes[1:])
        self.num_vars = sum((m+1)*n for m, n in self.weight_dims)

    def unpack_weights(self, z):
        """Unpack weight matrices and biases from a flattened vector."""
        for m, n in self.weight_dims:
            yield tf.reshape(z[:m*n],        [m, n]), \
                  tf.reshape(z[m*n:(m*n+n)], [1, n])
            z = z[(m+1)*n:]

    def mapping(self, x, z):
        """
        mu = NN(x; z)

        Note this is one sample of z at a time.

        Parameters
        -------
        x : tf.tensor
            n_data x D

        z : tf.tensor
            num_vars

        Returns
        -------
        tf.tensor
            vector of length n_data
        """
        h = x
        for W, b in self.unpack_weights(z):
            # broadcasting to do (h*W) + b (e.g. 40x10 + 1x10)
            h = self.nonlinearity(tf.matmul(h, W) + b)

        h = tf.squeeze(h) # n_data x 1 to n_data
        return h

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        # Data must have labels in the first column and features in
        # subsequent columns.
        y = xs[:, 0]
        x = xs[:, 1:]
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs, 1)
        mus = tf.pack([self.mapping(x, z) for z in tf.unpack(zs)])
        # broadcasting to do mus - y (n_minibatch x n_data - n_data)
        log_lik = -tf.reduce_sum(tf.pow(mus - y, 2), 1) / self.lik_variance
        return log_lik + log_prior

def build_toy_dataset(n_data=40, noise_std=0.1):
    ed.set_seed(0)
    D = 1
    x  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                         np.linspace(6, 8, num=n_data/2)])
    y = np.cos(x) + norm.rvs(0, noise_std, size=n_data)
    x = (x - 4.0) / 4.0
    x = x.reshape((n_data, D))
    y = y.reshape((n_data, 1))
    data = np.concatenate((y, x), axis=1) # n_data x (D+1)
    data = tf.constant(data, dtype=tf.float32)
    return ed.Data(data)

ed.set_seed(42)
model = BayesianNN(layer_sizes=[1, 10, 10, 1], nonlinearity=rbf)
variational = Variational()
variational.add(Normal(model.num_vars))
data = build_toy_dataset()

# Set up figure
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

inference = ed.MFVI(model, variational, data)
sess = inference.initialize(n_print=10)
for t in range(1000):
    loss = inference.update(sess)
    if t % inference.n_print == 0:
        print("iter {:d} loss {:.2f}".format(t, np.mean(loss)))

        # Sample functions from variational model
        mean, std = sess.run([variational.layers[0].m,
                              variational.layers[0].s])
        rs = np.random.RandomState(0)
        zs = rs.randn(10, variational.num_vars) * std + mean
        zs = tf.constant(zs, dtype=tf.float32)
        inputs = np.linspace(-8, 8, num=400, dtype=np.float32)
        x = tf.expand_dims(tf.constant(inputs), 1)
        mus = tf.pack([model.mapping(x, z) for z in tf.unpack(zs)])
        outputs = sess.run(mus)

        # Get data
        y, x = sess.run([data.data[:, 0], data.data[:, 1]])

        # Plot data and functions
        plt.cla()
        ax.plot(x, y, 'bx')
        ax.plot(inputs, outputs.T)
        ax.set_xlim([-8, 8])
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)
