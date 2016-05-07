import numpy as np
import tensorflow as tf

from edward.util import dot, get_dims, log_beta, log_gamma, multivariate_log_beta
from scipy import stats

class Distribution:
    """Template for all distributions."""
    def rvs(self, size=1):
        """
        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1

        Notes
        -----
        This is written in NumPy/SciPy, as TensorFlow does not support
        many distributions for random number generation.
        """
        raise NotImplementedError()

    def logpmf(self, x):
        """
        Arguments
        ---------
        x: np.array or tf.Tensor
        params: np.array or tf.Tensor

        Returns
        -------
        tf.Tensor
            scalar

        Note
        ----
        The following distributions use scalar arguments unless
        documented otherwise.
        """
        raise NotImplementedError()

class Bernoulli:
    def rvs(self, p, size=1):
        return stats.bernoulli.rvs(p, size=size)

    def logpmf(self, x, p):
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        p = tf.cast(tf.squeeze(p), dtype=tf.float32)
        return tf.mul(x, tf.log(p)) + tf.mul(1 - x, tf.log(1.0-p))

class Beta:
    def rvs(self, a, b, size=1):
        return stats.beta.rvs(a, b, size=size)

    def logpdf(self, x, a, b):
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        a = tf.cast(tf.squeeze(a), dtype=tf.float32)
        b = tf.cast(tf.squeeze(b), dtype=tf.float32)
        return (a-1) * tf.log(x) + (b-1) * tf.log(1-x) - log_beta(a, b)

class Dirichlet:
    def rvs(self, alpha, size=1):
        return stats.dirichlet.rvs(alpha, size=size)

    def logpdf(self, x, alpha):
        """
        Arguments
        ----------
        x: np.array or tf.Tensor
            vector
        alpha: np.array or tf.Tensor
            vector
        """
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        alpha = tf.cast(tf.squeeze(tf.convert_to_tensor(alpha)), dtype=tf.float32)
        return -multivariate_log_beta(alpha) + \
               tf.reduce_sum(tf.mul(alpha-1, tf.log(x)))

class Expon:
    def rvs(self, scale=1, size=1):
        return stats.expon.rvs(scale=scale, size=size)

    def logpdf(self, x, scale=1):
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        scale = tf.cast(tf.squeeze(scale), dtype=tf.float32)
        return - x/scale - tf.log(scale)

class Gamma:
    """Shape/scale parameterization"""
    def rvs(self, a, scale=1, size=1):
        return stats.gamma.rvs(a, scale=scale, size=size)

    def logpdf(self, x, a, scale=1):
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        a = tf.cast(tf.squeeze(a), dtype=tf.float32)
        scale = tf.cast(tf.squeeze(scale), dtype=tf.float32)
        return (a - 1.0) * tf.log(x) - x/scale - a * tf.log(scale) - log_gamma(a)

class InvGamma:
    """Shape/scale parameterization"""
    def rvs(self, alpha, scale=1, size=1):
        x = stats.invgamma.rvs(alpha, scale=scale, size=size)
        # This is temporary to avoid returning Inf values.
        x[np.logical_not(np.isfinite(x))] = 1.0
        return x

    def logpdf(self, x, alpha, scale=1):
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        alpha = tf.cast(tf.squeeze(alpha), dtype=tf.float32)
        scale = tf.cast(tf.squeeze(scale), dtype=tf.float32)
        return tf.mul(alpha, tf.log(scale)) - log_gamma(alpha) + \
               tf.mul(-alpha-1, tf.log(x)) - tf.truediv(scale, x)

class Multinomial:
    """There is no equivalent version implemented in SciPy."""
    def rvs(self, n, p, size=1):
        return np.random.multinomial(n, p, size=size)

    def logpmf(self, x, n, p):
        """
        Arguments
        ----------
        x: np.array or tf.Tensor
            vector of length K, where x[i] is the number of outcomes
            in the ith bucket
        n: int or tf.Tensor
            number of outcomes equal to sum x[i]
        p: np.array or tf.Tensor
            vector of probabilities summing to 1
        """
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        n = tf.cast(tf.squeeze(n), dtype=tf.float32)
        p = tf.cast(tf.squeeze(p), dtype=tf.float32)
        one = tf.constant(1.0, dtype=tf.float32)
        return log_gamma(n + one) - \
               tf.reduce_sum(log_gamma(x + one)) + \
               tf.reduce_sum(tf.mul(x, tf.log(p)))

class Multivariate_Normal:
    def rvs(self, mean=None, cov=1, size=1):
        return stats.multivariate_normal.rvs(mean, cov, size=size)

    def logpdf(self, x, mean=None, cov=1):
        """
        Arguments
        ----------
        x: np.array or tf.Tensor
            vector
        mean: np.array or tf.Tensor, optional
            vector. Defaults to zero mean.
        cov: np.array or tf.Tensor, optional
            vector or matrix. Defaults to identity.
        """
        x = tf.cast(tf.squeeze(tf.convert_to_tensor(x)), dtype=tf.float32)
        d = get_dims(x)[0]
        if mean is None:
            r = tf.ones([d]) * x
        else:
            mean = tf.cast(tf.squeeze(tf.convert_to_tensor(mean)), dtype=tf.float32)
            r = x - mean

        if cov is 1:
            cov_inv = tf.diag(tf.ones([d]))
            det_cov = tf.constant(1.0)
        else:
            cov = tf.cast(tf.squeeze(tf.convert_to_tensor(cov)), dtype=tf.float32)
            if len(cov.get_shape()) == 1:
                cov_inv = tf.diag(1.0 / cov)
                det_cov = tf.reduce_prod(cov)
            else:
                cov_inv = tf.matrix_inverse(cov)
                det_cov = tf.matrix_determinant(cov)
        r = tf.reshape(r, shape=(d, 1))
        lps = -0.5*d*tf.log(2*np.pi) - 0.5*tf.log(det_cov) - \
              0.5 * tf.matmul(tf.matmul(r, cov_inv, transpose_a=True), r)
        """
        # TensorFlow can't reverse-mode autodiff Cholesky
        L = tf.cholesky(cov)
        L_inv = tf.matrix_inverse(L)
        det_cov = tf.pow(tf.matrix_determinant(L), 2)
        inner = dot(L_inv, r)
        out = -0.5*d*tf.log(2*np.pi) - \
              0.5*tf.log(det_cov) - \
              0.5*tf.matmul(tf.transpose(inner), inner)
        """
        return tf.squeeze(lps)

    def entropy(self, mean=None, cov=1):
        """
        Note entropy does not depend on its mean.

        Arguments
        ----------
        mean: np.array or tf.Tensor, optional
            vector. Defaults to zero mean.
        cov: np.array or tf.Tensor, optional
            vector or matrix. Defaults to identity.
        """
        if cov is 1:
            d = 1
            det_cov = 1.0
        else:
            cov = tf.cast(tf.squeeze(tf.convert_to_tensor(cov)), dtype=tf.float32)
            d = get_dims(cov)[0]
            if len(cov.get_shape()) == 1:
                det_cov = tf.reduce_prod(cov)
            else:
                det_cov = tf.matrix_determinant(cov)

        return 0.5 * (d + d*tf.log(2*np.pi) + tf.log(det_cov))

class Norm:
    def rvs(self, loc=0, scale=1, size=1):
        return stats.norm.rvs(loc, scale, size=size)

    def logpdf(self, x, loc=0, scale=1):
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        loc = tf.cast(tf.squeeze(loc), dtype=tf.float32)
        scale = tf.cast(tf.squeeze(scale), dtype=tf.float32)
        z = (x - loc) / scale
        return -0.5*tf.log(2*np.pi) - tf.log(scale) - 0.5*z*z

    def entropy(self, loc=0, scale=1):
        """Note entropy does not depend on its mean."""
        scale = tf.cast(tf.squeeze(scale), dtype=tf.float32)
        return 0.5 * (1 + tf.log(2*np.pi) + tf.log(scale*scale))

class Poisson:
    def rvs(self, mu, size=1):
        return stats.poisson.rvs(mu, size=size)

    def logpmf(self, x, mu):
        x = tf.squeeze(x)
        mu = tf.cast(tf.squeeze(mu), dtype=tf.float32)
        return x * tf.log(mu) - mu - log_gamma(x + 1.0)

class T:
    def rvs(self, df, loc=0, scale=1, size=1):
        return stats.t.rvs(df, loc=loc, scale=scale, size=size)

    def logpdf(self, x, df, loc=0, scale=1):
        x = tf.cast(tf.squeeze(x), dtype=tf.float32)
        df = tf.squeeze(df)
        loc = tf.cast(tf.squeeze(loc), dtype=tf.float32)
        scale = tf.cast(tf.squeeze(scale), dtype=tf.float32)
        return 0.5 * log_gamma(df + 1.0) - \
               log_gamma(0.5 * df) - \
               0.5 * (np.log(np.pi) + tf.log(df)) +  tf.log(scale) - \
               0.5 * (df + 1.0) * \
                   tf.log(1.0 + (1.0/df) * tf.square((x-loc)/scale))

class TruncNorm:
    def rvs(self, a, b, loc=0, scale=1, size=1):
        return stats.truncnorm.rvs(a, b, loc, scale, size=size)

    def logpdf(self, a, b, loc=0, scale=1):
        cdf = stats.norm.cdf
        cst = cdf((b - loc)/scale) - cdf((a - loc)/scale)
        cst = -np.log(scale) - np.log(cst)
        return cst + norm.logpdf(loc, scale)

class Wishart:
    def rvs(self, df, scale, size=1):
        return stats.wishart.rvs(df, scale, size=size)

    def logpdf(self, x, df, scale):
        raise NotImplementedError()

bernoulli = Bernoulli()
beta = Beta()
dirichlet = Dirichlet()
expon = Expon() # TODO unit test
gamma = Gamma()
invgamma = InvGamma()
multinomial = Multinomial()
multivariate_normal = Multivariate_Normal()
norm = Norm()
poisson = Poisson() # TODO unit test
t = T() # TODO unit test
truncnorm = TruncNorm() # TODO unit test
wishart = Wishart() # TODO unit test
