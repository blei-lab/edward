import numpy as np
import tensorflow as tf

from blackbox.util import log_multinomial, log_inv_gamma, log_dirichlet, log_beta, log_gamma, dot, get_dims
from scipy import stats

class Bernoulli:
    def rvs(self, p, size=1):
        """Written in NumPy/SciPy."""
        return stats.bernoulli.rvs(p, size=size)

    def logpmf(self, x, p):
        """Written in TensorFlow."""
        return tf.mul(x, tf.log(p)) + tf.mul(1 - x, tf.log(1.0-p))

class Beta:
    def rvs(self, a, b, size=1):
        """Written in NumPy/SciPy."""
        return stats.beta.rvs(a, b, size=size)

    def logpdf(self, x, a, b):
        """Written in TensorFlow."""
        return (a-1) * tf.log(x) + (b-1) * tf.log(1-x) - log_beta(a, b)

class Dirichlet:
    def rvs(self, alpha, size=1):
        """Written in NumPy/SciPy."""
        return stats.dirichlet.rvs(alpha, size=size)

    def logpdf(self, x, alpha):
        """Written in TensorFlow."""
        log_dir = log_dirichlet(alpha)
        return -log_dir + tf.reduce_sum(tf.mul(alpha-1, tf.log(x)))

class Expon:
    def rvs(self, scale=1, size=1):
        """Written in NumPy/SciPy."""
        return stats.expon.rvs(scale=scale, size=size)

    def logpdf(self, x, scale=1):
        """Written in TensorFlow."""
        return - x/scale - tf.log(scale)

class Gamma:
    """This is the shape/scale parameterization of the gamma distribution"""
    def rvs(self, a, scale=1, size=1):
        """Written in NumPy/SciPy."""
        return stats.gamma.rvs(a, scale=scale, size=size)

    def logpdf(self, x, a, scale=1):
        """Written in TensorFlow."""
        return (a - 1.0) * tf.log(x) - x/scale - a * tf.log(scale) - log_gamma(a)

class InvGamma:
    def rvs(self, alpha, beta, size=1):
        """Written in NumPy/SciPy."""
        return stats.invgamma.rvs(alpha, scale=beta, size=size)

    def logpdf(self, x, alpha, beta):
        """Written in TensorFlow."""
        return -log_inv_gamma(alpha, beta) - \
               tf.mul(alpha+1, tf.log(x)) - tf.truediv(beta, x)

class Multinomial:
    def rvs(self, n, p, size=1):
        """Written in NumPy/SciPy."""
        return np.random.multinomial(n, p, size=size)

    def logpmf(self, x, n, p):
        """Written in TensorFlow."""
        return -log_multinomial(x, n) + tf.reduce_sum(tf.mul(x, tf.log(p)))

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

class Poisson:
    def rvs(self, mu, size=1):
        """Written in NumPy/SciPy."""
        return stats.poisson.rvs(mu, size=size)

    def logpmf(self, x, mu):
        """Written in TensorFlow."""
        return x * tf.log(mu) - mu - log_gamma(x + 1.0)

class T:
    def rvs(self, df, loc=0, scale=1, size=1):
        """Written in NumPy/SciPy."""
        return stats.t.rvs(df, loc=loc, scale=scale, size=size)

    def logpdf(self, x, df, loc=0, scale=1):
        """Written in TensorFlow."""
        return 0.5 * log_gamma(df + 1.0) - \
               log_gamma(0.5 * df) - \
               0.5 * (np.log(np.pi) + tf.log(df)) +  tf.log(scale) - \
               0.5 * (df + 1.0) * \
                   tf.log(1.0 + (1.0/df) * tf.square((x-loc)/scale))

class Wishart:
    def rvs(self, df, scale, size=1):
        """Written in NumPy/SciPy."""
        return stats.wishart.rvs(df, scale, size=size)

    def logpdf(self, x, df, scale):
        """Written in TensorFlow."""
        raise NotImplementedError()

bernoulli = Bernoulli()
beta = Beta()
dirichlet = Dirichlet()
expon = Expon()
gamma = Gamma()
invgamma = InvGamma()
multinomial = Multinomial()
norm = Norm()
poisson = Poisson()
t = T()
wishart = Wishart()
