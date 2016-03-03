import numpy as np
import tensorflow as tf

from blackbox.util import log_beta, dot, get_dims
from scipy import stats

class Bernoulli:
    def rvs(self, p, size):
        """Written in NumPy/SciPy."""
        return stats.bernoulli.rvs(p, size=size)

    def logpmf(self, x, p):
        """Written in TensorFlow."""
        lp = tf.to_float(tf.log(p))
        lp1 = tf.to_float(tf.log(1.0 - p))
        return tf.mul(x, lp) + tf.mul(1 - x, lp1)

class Beta:
    def rvs(self, a, b, size):
        """Written in NumPy/SciPy."""
        return stats.beta.rvs(a, b, size=size)

    def logpdf(self, x, a, b):
        """Written in TensorFlow."""
        return (a-1) * tf.log(x) + (b-1) * tf.log(1-x) - log_beta(a, b)

class Norm:
    def rvs(self, loc=0, scale=1, size=1):
        """Written in NumPy/SciPy."""
        return stats.norm.rvs(loc, scale, size=size)

    def logpdf(self, x, mu=None, Sigma=None):
        """
        Written in TensorFlow.

        Arguments
        ----------
        x: Tensor scalar, vector
        mu: mean - None, Tensor scalar, vector
        Sigma: variance - None, Tensor scalar, vector, matrix
        TODO allow minibatch
        TODO this doesn't do any error checking
        """
        d = get_dims(x)[0]
        if mu is None:
            r = tf.ones([d]) * x
        elif len(mu.get_shape()) == 0: # scalar
            r = x - mu
        else:
            r = tf.sub(x, mu)

        if Sigma is None:
            Sigma_inv = tf.diag(tf.ones([d]))
            det_Sigma = tf.constant(1.0)
        elif len(Sigma.get_shape()) == 0: # scalar
            Sigma_inv = 1.0 / Sigma
            det_Sigma = Sigma
        elif len(Sigma.get_shape()) == 1: # vector
            Sigma_inv = tf.diag(1.0 / Sigma)
            det_Sigma = tf.reduce_prod(Sigma)
        else:
            Sigma_inv = tf.matrix_inverse(Sigma)
            det_Sigma = tf.matrix_determinant(Sigma)

        if d == 1:
            lps = -0.5*d*tf.log(2*np.pi) - \
                  0.5*tf.log(det_Sigma) - \
                  0.5*r * Sigma_inv * r
        else:
            lps = -0.5*d*tf.log(2*np.pi) - \
                  0.5*tf.log(det_Sigma) - \
                  0.5*dot(dot(tf.transpose(r), Sigma_inv), r)
            """
            # TensorFlow can't reverse-mode autodiff Cholesky
            L = tf.cholesky(Sigma)
            L_inv = tf.matrix_inverse(L)
            det_Sigma = tf.pow(tf.matrix_determinant(L), 2)
            inner = dot(L_inv, r)
            out = -0.5*d*tf.log(2*np.pi) - \
                  0.5*tf.log(det_Sigma) - \
                  0.5*tf.matmul(tf.transpose(inner), inner)
            """
        #lp = tf.reduce_sum(lps)
        #return lps
        return tf.reshape(lps, [-1])

    def entropy(self, Sigma):
        """
        - E_{Gaussian(x; mu, Sigma)} [ Gaussian(x; mu, Sigma) ]
        Note that the entropy of a Gaussian does not depend on its mean.

        Arguments
        ----------
        Sigma: variance - Tensor scalar, vector, matrix
        """
        d = get_dims(Sigma)[0]
        if len(Sigma.get_shape()) == 0: # scalar
            det_Sigma = Sigma
        elif len(Sigma.get_shape()) == 1: # vector
            det_Sigma = tf.reduce_prod(Sigma)
        else:
            det_Sigma = tf.matrix_determinant(Sigma)

        return 0.5 * (d + d*np.log(2*np.pi) + tf.log(det_Sigma))

bernoulli = Bernoulli()
beta = Beta()
norm = Norm()
