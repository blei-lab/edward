#!/usr/bin/env python
"""
Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
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
from edward.util import rbf


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
  def __init__(self, layer_sizes, nonlinearity=tf.tanh,
               lik_std=0.1, prior_std=1.0):
    self.layer_sizes = layer_sizes
    self.nonlinearity = nonlinearity
    self.lik_std = lik_std
    self.prior_std = prior_std

    self.n_layers = len(layer_sizes)
    self.weight_dims = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    self.n_vars = sum((m + 1) * n for m, n in self.weight_dims)

  def unstack_weights(self, zs):
    """Unstack weight matrices and biases from a flattened vector."""
    for m, n in self.weight_dims:
      yield tf.reshape(zs[:(m * n)], [m, n]), \
          tf.reshape(zs[(m * n):(m * n + n)], [n])
      zs = zs[(m + 1) * n:]

  def neural_network(self, x, zs):
    """Forward pass of the neural net, outputting a vector of
    `n_minibatch` elements."""
    h = x
    for W, b in self.unstack_weights(zs):
      h = self.nonlinearity(tf.matmul(h, W) + b)

    return tf.squeeze(h)  # n_minibatch x 1 to n_minibatch

  def log_lik(self, xs, zs):
    """Return scalar, the log-likelihood p(xs | zs)."""
    x, y = xs['x'], xs['y']
    mu = self.neural_network(x, zs['z'])
    log_lik = tf.reduce_sum(norm.logpdf(y, mu, self.lik_std))
    return log_lik


def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
  x = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = np.cos(x) + np.random.normal(0, noise_std, size=N)
  x = (x - 4.0) / 4.0
  x = x.reshape((N, D))
  return x, y


ed.set_seed(42)
x_train, y_train = build_toy_dataset()

model = BayesianNN(layer_sizes=[1, 10, 10, 1], nonlinearity=rbf)

qz_mu = tf.Variable(tf.random_normal([model.n_vars]))
qz_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([model.n_vars])))
qz = Normal(mu=qz_mu, sigma=qz_sigma)

# Set up figure
fig = plt.figure(figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

# model.log_lik() is defined so KLqp will do variational inference
# assuming a standard normal prior on the weights; this enables VI
# with an analytic KL term which provides faster inference.
sess = ed.get_session()
data = {'x': x_train, 'y': y_train}
inference = ed.KLqp({'z': qz}, data, model)
inference.initialize(n_print=10)

init = tf.global_variables_initializer()
init.run()

for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  if t % inference.n_print == 0:
    # Sample functions from variational model
    mean, std = sess.run([qz.mu, qz.sigma])
    rs = np.random.RandomState(0)
    zs = rs.randn(10, model.n_vars) * std + mean
    zs = tf.convert_to_tensor(zs, dtype=tf.float32)
    inputs = np.linspace(-8, 8, num=400, dtype=np.float32)
    x = tf.expand_dims(inputs, 1)
    mus = []
    for z in tf.unstack(zs):
      mus += [model.neural_network(x, z)]

    outputs = tf.stack(mus).eval()

    # Get data
    x, y = data['x'], data['y']

    # Plot data and functions
    plt.cla()
    ax.plot(x, y, 'bx')
    ax.plot(inputs, outputs.T)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-2, 3])
    plt.draw()
    plt.pause(1.0 / 60.0)
