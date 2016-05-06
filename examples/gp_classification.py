#!/usr/bin/env python
"""
Gaussian process classification using mean-field variational inference.

Probability model:
    Gaussian process classification
    Prior: Gaussian process with RBF kernel
    Likelihood: Bernoulli-Probit or Bernoulli-Logit
Variational model
    Likelihood: Mean-field Gaussian
"""
import edward as ed
import tensorflow as tf
import numpy as np

from edward.stats import bernoulli, multivariate_normal
from edward.util import sigmoid

class GaussianProcess:
    """
    Gaussian process classification with known kernel hyperparameters.

    Parameters
    ----------
    N : int
        Number of data points.
    eta : float, optional
        Variance.
    l : float, optional
        Length scale.
    """
    def __init__(self, N, eta=1.0, l=1.0):
        self.N = N
        self.eta = eta
        self.l = l

        self.num_vars = N
        self.inverse_link = sigmoid

    def kernel_xy(self, x, xstar, sess):
        """Covariance matrix for a pair of points k(x, xstar)."""
        # TODO use an rbf function from edward.util
        x = sess.run(x)
        xstar = sess.run(xstar)
        s = np.power(self.eta, 2.0) * \
            np.exp(-1.0/(np.power(self.l,   2.0) * 2.0) * \
                         (np.sum(np.power(x - xstar , 2.0))))

        return s

    def kernel(self, xs):
        """Compute covariance matrix."""
        # TODO fill in kernel without invoking tf.Session()
        sess = tf.Session()
        mat = np.zeros(shape=(self.N, self.N))
        for i in xrange(self.N):
            xi = xs[i, 1:]
            for j in xrange(self.N):
                if j == i:
                    mat[i, i] = self.kernel_xy(xi, xi, sess)
                else:
                    sij = self.kernel_xy(xi, xs[j, 1:], sess)
                    mat[i, j] = sij

        sess.close()
        return tf.constant(mat, dtype=tf.float32)

    def log_prob(self, xs, zs):
        K = self.kernel(xs)
        log_prior = multivariate_normal.logpdf(zs[:, :], cov=K)
        log_lik = tf.pack([tf.reduce_sum( \
            bernoulli.logpmf(xs[:,0], self.inverse_link(tf.mul(xs[:,0], z))) \
            ) for z in tf.unpack(zs)])
        return log_prior + log_lik

ed.set_seed(42)
# Data must have labels in the first column and features in
# subsequent columns.
df = np.loadtxt('data/crabs_train.txt', dtype='float32', delimiter=',')
data = ed.Data(tf.constant(df, dtype=tf.float32))

model = GaussianProcess(N=len(df))
variational = ed.MFGaussian(model.num_vars)
inference = ed.MFVI(model, variational, data)
inference.run(n_iter=10000)
