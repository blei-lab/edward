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

from edward.models import Normal
from edward.stats import norm

plt.style.use('ggplot')


class BayesianNN:
  """
  Bayesian neural network for regressing outputs y on inputs x.

  p((x,y), z) = Normal(y | NN(x; z), lik_std) *
                Normal(z | 0, prior_std),

  where z are neural network weights, and with known likelihood and
  prior standard deviations.

  Parameters
  ----------
  layer_sizes : list
    The size of each layer, ordered from input to output.
  nonlinearity : function, optional
    Non-linearity after each linear transformation in the neural
    network; aka activation function.
  lik_std : float, optional
    Standard deviation of the normal likelihood; aka noise parameter,
    homoscedastic noise, scale parameter.
  prior_std : float, optional
    Standard deviation of the normal prior on weights; aka L2
    regularization parameter, ridge penalty, scale parameter.
  """
  def __init__(self, layer_sizes, nonlinearity=tf.nn.tanh,
               lik_std=0.1, prior_std=1.0):
    self.layer_sizes = layer_sizes
    self.nonlinearity = nonlinearity
    self.lik_std = lik_std
    self.prior_std = prior_std

    self.n_layers = len(layer_sizes)
    self.weight_dims = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    self.n_vars = sum((m + 1) * n for m, n in self.weight_dims)

  def unpack_weights(self, zs):
    """Unpack weight matrices and biases from a flattened vector."""
    for m, n in self.weight_dims:
      yield tf.reshape(zs[:(m * n)], [m, n]), \
          tf.reshape(zs[(m * n):(m * n + n)], [n])
      zs = zs[(m + 1) * n:]

  def neural_network(self, x, zs):
    """Forward pass of the neural net, outputting a vector of
    `n_minibatch` elements."""
    h = x
    for W, b in self.unpack_weights(zs):
      h = self.nonlinearity(tf.matmul(h, W) + b)

    return tf.squeeze(h)  # n_minibatch x 1 to n_minibatch

  def log_lik(self, xs, zs):
    """Return scalar, the log-likelihood p(xs | zs)."""
    x, y = xs['x'], xs['y']
    mu = self.neural_network(x, zs['z'])
    log_lik = tf.reduce_sum(norm.logpdf(y, mu, self.lik_std))
    return log_lik


def build_toy_dataset(N=50, noise_std=0.1):
  x = np.linspace(-3, 3, num=N)
  y = np.cos(x) + norm.rvs(0, noise_std, size=N)
  x = x.reshape((N, 1))
  return x, y


ed.set_seed(42)
x_train, y_train = build_toy_dataset()

model = BayesianNN(layer_sizes=[1, 2, 2, 1])

qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

data = {'x': x_train, 'y': y_train}
inference = ed.MFVI({'z': qz}, data, model)
inference.initialize()


sess = ed.get_session()

# FIRST VISUALIZATION (prior)

# Sample functions from variational model
mean, std = sess.run([qz.mu, qz.sigma])
rs = np.random.RandomState(0)
zs = rs.randn(10, model.n_vars) * std + mean
zs = tf.convert_to_tensor(zs, dtype=tf.float32)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(tf.constant(inputs), 1)
mus = []
for z in tf.unpack(zs):
  mus += [model.neural_network(x, z)]

outputs = tf.pack(mus).eval()
x, y = data['x'], data['y']

# Plot data and functions
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 0 - (CLOSE WINDOW TO CONTINUE)")
ax.plot(x, y, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()


# RUN MEAN-FIELD VARIATIONAL INFERENCE
inference.run(n_iter=1000, n_samples=5, n_print=100)


# SECOND VISUALIZATION (posterior)

# Sample functions from variational model
mean, std = sess.run([qz.mu, qz.sigma])
rs = np.random.RandomState(0)
zs = rs.randn(10, model.n_vars) * std + mean
zs = tf.convert_to_tensor(zs, dtype=tf.float32)
inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
x = tf.expand_dims(tf.constant(inputs), 1)
mus = []
for z in tf.unpack(zs):
  mus += [model.neural_network(x, z)]

outputs = tf.pack(mus).eval()
x, y = data['x'], data['y']

# Plot data and functions
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Iteration: 1000 - (CLOSE WINDOW TO TERMINATE)")
ax.plot(x, y, 'ks', alpha=0.5, label='(x, y)')
ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
ax.set_xlim([-5, 5])
ax.set_ylim([-2, 2])
ax.legend()
plt.show()
