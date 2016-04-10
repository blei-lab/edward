#!/usr/bin/env python
"""
Mean-field variational inference on a Bayesian neural network.
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016))
Inspired by autograd's Bayesian neural network example.

Probability model:
    Bayesian neural network
    Prior: Normal
    Likelihood: Normal with mean parameterized by fully connected NN
Variational model
    Likelihood: Mean-field Gaussian
"""
import edward as ed
import tensorflow as tf
import numpy as np

from edward.stats import norm
from edward.util import get_dims, rbf

class BayesianNN:
    """
    Bayesian neural network for regressing outputs y on inputs x.

    p((x,y), z) = Normal(y; NN(x; z), lik_variance) *
                  Normal(z; 0, prior_variance),

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
        Variance of the normal prior on neural network weights; aka L2
        regularization parameter, ridge penalty, scale parameter.
    """
    def __init__(self, layer_sizes, nonlinearity=tf.nn.tanh,
        lik_variance=0.01, prior_variance=0.01):
        self.layer_sizes = layer_sizes
        self.nonlinearity = nonlinearity
        self.lik_variance = lik_variance
        self.prior_variance = prior_variance

        self.num_layers = len(layer_sizes)
        self.weight_dims = zip(layer_sizes[:-1], layer_sizes[1:])
        self.num_vars = sum((m+1)*n for m, n in self.weight_dims)

    def unpack_weights(self, zs):
        """Unpack weight matrices and biases from a flattened vector."""
        #n_minibatch = len(zs) # TODO
        for m, n in self.weight_dims:
            # TODO can't subset tf.tensors(?)
            # i would reshape it into a
            # tf.tensor of
            # [matrix/vector, ..., ] x n_minibatch x (m or 1) x n
            #yield zs[:, :m*n]     .reshape((n_minibatch, m, n)), \
            #      zs[:, m*n:m*n+n].reshape((n_minibatch, 1, n))
            #zs = zs[:, (m+1)*n:]
            yield tf.reshape(zs[:m*n],      [m, n]), \
                  tf.reshape(zs[m*n:m*n+n], [1, n])
            zs = zs[(m+1)*n:]

    def mapping(self, x, zs):
        """
        mu = NN(x; z)
        A vector for z in zs.
        """
        #h = tf.expand_dims(x, 0) # TODO
        # using batch_matmul
        zs = zs[0, :]
        h = x
        for W, b in self.unpack_weights(zs):
            # broadcasting to do (W*h) + b (e.g. 40x10 + 1x10)
            h = self.nonlinearity(tf.matmul(h, W) + b)

        return h

    def log_prob(self, xs, zs):
        """
        Calculates the unnormalized log joint density.

        Parameters
        ----------
        xs : tuple
            tuple of inputs, a np.ndarray of dimension n_data x D,
            and outputs, a np.ndarray of dimension n_data x D
        zs : tf.tensor or np.ndarray
            n_minibatch x num_vars, where n_minibatch is the number of
            weight samples and num_vars is the number of weights

        Returns
        -------
        tf.tensor
            n_minibatch array where the i^th element is the log joint
            density of x and zs[i, :]
        """
        x, y = xs
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs, 1)
        mu = self.mapping(x, zs)
        log_lik = -tf.reduce_sum(tf.pow(y - mu, 2), 1) / self.lik_variance
        return log_lik + log_prior

def build_toy_dataset(n_data=40, noise_std=0.1):
    ed.set_seed(0)
    D = 1
    x  = np.concatenate([np.linspace(0, 2, num=n_data/2, dtype=np.float32),
                         np.linspace(6, 8, num=n_data/2, dtype=np.float32)])
    y = np.cos(x) + norm.rvs(0, noise_std, size=n_data)
    x = (x - 4.0) / 4.0
    x = x.reshape((n_data, D))
    y = y.reshape((n_data, D))
    return ed.Data((x, y))

ed.set_seed(42)
model = BayesianNN(layer_sizes=[1, 10, 10, 1], nonlinearity=rbf)
variational = ed.MFGaussian(model.num_vars)
data = build_toy_dataset()

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=1000)

# TODO visualization; see autograd
# https://github.com/HIPS/autograd/blob/master/examples/bayesian_neural_net.py
# https://www.youtube.com/watch?v=xrCalU-MPCc
