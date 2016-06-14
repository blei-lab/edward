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
from edward.util import get_dims
from edward.models import Variational, Normal
from edward.stats import bernoulli, norm

class ProbitReg:
    """
    Bayesian probit  regression for outputs y on inputs x.

    p((x,y), z) = Bernoulli(y | probit(x*z / sigma)) *
                  Normal(z | 0, prior_variance),

    where z are weights, sigma is the variance of latent gaussian variables \phi such that 
    \phi ~ N(x*z, sigma) and y = 1(\phi > 0)
    
    Parameters
    ----------
    weight_dim : list
        Dimension of weights, which is dimension of input x dimension
        of output.
    sigma : float, optional
        Variance of the latent variables \phi 
    prior_variance : float, optional
        Variance of the normal prior on weights; aka L2
        regularization parameter, ridge penalty, scale parameter.
    """
    def __init__(self,  weight_dim, prior_variance=20.0, sigma=5.0):
        self.weight_dim = weight_dim
        self.prior_variance = prior_variance
        self.sigma = sigma
        self.num_vars = (self.weight_dim[0])* (self.weight_dim[1]+1)

    def probit(self, x):
        return 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))

    def mapping(self, x, z):
        """
        Inverse link function on linear transformation,
        probit((W*x + b)/sigma)
        """
        n, D = self.weight_dim[0], self.weight_dim[1]
        W = tf.reshape(z[:D*n], [D, n])
        b = tf.reshape(z[D*n:], [1, n])
        h = self.probit((tf.matmul(x, W) + b) / self.sigma)
        h = tf.squeeze(h) 
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
        log_lik = tf.reduce_sum(log_lik, 1)
        log_prior = -0.5*tf.log(2*np.pi) - 0.5 * tf.log(self.prior_variance) -\
            (0.5 / self.prior_variance) * tf.reduce_sum(zs*zs, 1)
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
    return ed.Data(data), D

ed.set_seed(42)
data, D = build_toy_dataset()
n_minibatch = 1
model = ProbitReg(weight_dim=[n_minibatch,D])
variational = Variational()
variational.add(Normal(model.num_vars))


# Set up figure
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

inference = ed.MFVI(model, variational, data)
inference.initialize(n_print=5)
sess = ed.get_session()
for t in range(1000):
    loss = inference.update()
    if t % inference.n_print == 0:
        print("iter {:d} loss {:.2f}".format(t, loss))
        variational.print_params()

        # Sample functions from variational model
        mean, std = sess.run([variational.layers[0].m,
                              variational.layers[0].s])
        rs = np.random.RandomState(0)
        zs = rs.randn(10, variational.num_vars) * std + mean
        zs = tf.constant(zs, dtype=tf.float32)
        inputs = np.linspace(-3, 3, num=400, dtype=np.float32)
        x = tf.expand_dims(tf.constant(inputs), 1)
        mus = tf.pack([model.mapping(x, z) for z in tf.unpack(zs)])
        outputs = mus.eval()

        # Get data
        y, x = sess.run([data.data[:, 0], data.data[:, 1]])

        # Plot data and functions
        plt.cla()
        ax.plot(x, y, 'bx')
        ax.plot(inputs, outputs.T)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-0.5, 1.5])
        plt.draw()
        plt.pause(1.0/60.0)
