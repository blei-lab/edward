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
plt.style.use('ggplot')
import numpy as np

from edward.models import Variational, Normal
from edward.stats import norm

class BayesianNN:
    """
    Bayesian neural network for regressing outputs y on inputs x.

    p((x,y), z) = Normal(y | NN(x; z), lik_variance) *
                  Normal(z | 0, 1),

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
    """
    def __init__(self, layer_sizes, nonlinearity=tf.nn.tanh,
        lik_variance=0.01):
        self.layer_sizes = layer_sizes
        self.nonlinearity = nonlinearity
        self.lik_variance = lik_variance

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
        """mu = NN(x; z)

        This is one sample of z at a time.

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

    def log_lik(self, xs, zs):
        """Returns a vector [log p(xs | zs[1,:]), ..., log p(xs | zs[S,:])]."""
        # Data must have labels in the first column and features in
        # subsequent columns.
        y = xs[:, 0]
        x = xs[:, 1:]
        mus = tf.pack([self.mapping(x, z) for z in tf.unpack(zs)])
        log_lik = tf.reduce_sum(norm.logpdf(y, loc=mus, scale=self.lik_variance), 1)
        return log_lik

def build_toy_dataset(n_data=50, noise_std=0.1):
    x = np.linspace(-3, 3, num=n_data)
    y = np.cos(x) 
    y = y + norm.rvs(0, noise_std, size=n_data).reshape((n_data,))
    x = x.reshape((n_data, 1))
    y = y.reshape((n_data, 1))
    data = np.concatenate((y, x), axis=1) # n_data x 2
    data = tf.constant(data, dtype=tf.float32)
    return ed.Data(data)    

ed.set_seed(42)

model = BayesianNN(layer_sizes=[1, 2, 2, 1])

variational = Variational()
variational.add(Normal(model.num_vars))

data = build_toy_dataset()

inference = ed.MFVI(model, variational, data)
inference.initialize()


sess = ed.get_session()

# FIRST VISUALIZATION (prior)

# Sample functions from variational model
mean, std = sess.run([variational.layers[0].loc,
                      variational.layers[0].scale])
rs = np.random.RandomState(0)
zs = rs.randn(10, variational.num_vars) * std + mean
zs = tf.constant(zs, dtype=tf.float32)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(tf.constant(inputs), 1)
mus = tf.pack([model.mapping(x, z) for z in tf.unpack(zs)])
outputs = mus.eval()
y, x = sess.run([data.data[:, 0], data.data[:, 1]])

# Plot data and functions
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0 - (CLOSE WINDOW TO CONTINUE)")
ax.plot(x, y, 'ks', alpha=0.5, label='y')
ax.plot(x, x, 'k-', lw=2, label='x')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()


# RUN MEAN-FIELD VARIATIONAL INFERENCE
inference.run(n_iter=1000, n_minibatch=5, n_print=100)


# SECOND VISUALIZATION (posterior)

# Sample functions from variational model
mean, std = sess.run([variational.layers[0].loc,
                      variational.layers[0].scale])
rs = np.random.RandomState(0)
zs = rs.randn(10, variational.num_vars) * std + mean
zs = tf.constant(zs, dtype=tf.float32)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(tf.constant(inputs), 1)
mus = tf.pack([model.mapping(x, z) for z in tf.unpack(zs)])
outputs = mus.eval()
y, x = sess.run([data.data[:, 0], data.data[:, 1]])

# Plot data and functions
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 1000 - (CLOSE WINDOW TO TERMINATE)")
ax.plot(x, y, 'ks', alpha=0.5, label='y')
ax.plot(x, x, 'k-', lw=2, label='x')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()
