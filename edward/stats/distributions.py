import numpy as np
import tensorflow as tf

from edward.util import dot, get_dims, digamma, lbeta, lgamma
from itertools import product
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
        Parameters
        ---------
        x : np.array or tf.Tensor
            If univariate distribution, can be a scalar, vector, or matrix.
            If multivariate distribution, can be a vector or matrix.

        params : np.array or tf.Tensor
            scalar unless documented otherwise

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.
        """
        raise NotImplementedError()

    def entropy(self):
        """
        Parameters
        ---------
        params : np.array or tf.Tensor
            If univariate distribution, can be a scalar or vector.
            If multivariate distribution, can be a vector or matrix.

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar or vector
            corresponding to the size of input. For multivariate
            distributions, returns a scalar if vector input and vector
            if matrix input, where each element in the vector
            evaluates a row in the matrix.

        Notes
        -----
        SciPy doesn't always enable vector inputs for
        univariate distributions or matrix inputs for multivariate
        distributions. This does.
        """
        raise NotImplementedError()

class Bernoulli:
    def rvs(self, p, size=1):
        return stats.bernoulli.rvs(p, size=size)

    def logpmf(self, x, p):
        x = tf.cast(x, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.mul(x, tf.log(p)) + tf.mul(1.0 - x, tf.log(1.0-p))

    def entropy(self, p):
        p = tf.cast(p, dtype=tf.float32)
        return -tf.mul(p, tf.log(p)) - tf.mul(1.0 - p, tf.log(1.0-p))

class Beta:
    def rvs(self, a, b, size=1):
        return stats.beta.rvs(a, b, size=size)

    def logpdf(self, x, a, b):
        x = tf.cast(x, dtype=tf.float32)
        a = tf.cast(tf.squeeze(a), dtype=tf.float32)
        b = tf.cast(tf.squeeze(b), dtype=tf.float32)
        return (a-1) * tf.log(x) + (b-1) * tf.log(1-x) - lbeta(tf.pack([a, b]))

    def entropy(self, a, b):
        a = tf.cast(tf.squeeze(a), dtype=tf.float32)
        b = tf.cast(tf.squeeze(b), dtype=tf.float32)
        if len(a.get_shape()) == 0:
            return lbeta(tf.pack([a, b])) - \
                   tf.mul(a - 1.0, digamma(a)) - \
                   tf.mul(b - 1.0, digamma(b)) + \
                   tf.mul(a + b - 2.0, digamma(a+b))
        else:
            return lbeta(tf.concat(1,
                         [tf.expand_dims(a, 1), tf.expand_dims(b, 1)])) - \
                   tf.mul(a - 1.0, digamma(a)) - \
                   tf.mul(b - 1.0, digamma(b)) + \
                   tf.mul(a + b - 2.0, digamma(a+b))

class Binom:
    def rvs(self, n, p, size=1):
        return stats.binom.rvs(p, size=size)

    def logpmf(self, x, n, p):
        x = tf.cast(x, dtype=tf.float32)
        n = tf.cast(n, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return lgamma(n + 1.0) - lgamma(x + 1.0) - lgamma(n - x + 1.0) + \
               tf.mul(x, tf.log(p)) + tf.mul(n - x, tf.log(1.0-p))

    def entropy(self, n, p):
        raise NotImplementedError()

class Chi2:
    def rvs(self, df, size=1):
        return stats.chi2.rvs(df, size=size)

    def logpdf(self, x, df):
        x = tf.cast(x, dtype=tf.float32)
        df = tf.cast(df, dtype=tf.float32)
        return tf.mul(0.5*df - 1, tf.log(x)) - 0.5*x - \
               tf.mul(0.5*df, tf.log(2.0)) - lgamma(0.5*df)

    def entropy(self, df):
        raise NotImplementedError()

class Dirichlet:
    def rvs(self, alpha, size=1):
        return stats.dirichlet.rvs(alpha, size=size)

    def logpdf(self, x, alpha):
        """
        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        alpha : np.array or tf.Tensor
            vector
        """
        x = tf.cast(x, dtype=tf.float32)
        alpha = tf.cast(tf.convert_to_tensor(alpha), dtype=tf.float32)
        if len(get_dims(x)) == 1:
            return -lbeta(alpha) + tf.reduce_sum(tf.mul(alpha-1, tf.log(x)))
        else:
            return -lbeta(alpha) + tf.reduce_sum(tf.mul(alpha-1, tf.log(x)), 1)

    def entropy(self, alpha):
        """
        Arguments
        ----------
        alpha: np.array or tf.Tensor
            vector or matrix
        """
        alpha = tf.cast(tf.convert_to_tensor(alpha), dtype=tf.float32)
        if len(get_dims(alpha)) == 1:
            K = get_dims(alpha)[0]
            a = tf.reduce_sum(alpha)
            return lbeta(alpha) + \
                   tf.mul(a - K, digamma(a)) - \
                   tf.reduce_sum(tf.mul(alpha-1, digamma(alpha)))
        else:
            K = get_dims(alpha)[1]
            a = tf.reduce_sum(alpha, 1)
            return lbeta(alpha) + \
                   tf.mul(a - K, digamma(a)) - \
                   tf.reduce_sum(tf.mul(alpha-1, digamma(alpha)), 1)

class Expon:
    def rvs(self, scale=1, size=1):
        return stats.expon.rvs(scale=scale, size=size)

    def logpdf(self, x, scale=1):
        x = tf.cast(x, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return - x/scale - tf.log(scale)

    def entropy(self, scale=1):
        raise NotImplementedError()

class Gamma:
    """Shape/scale parameterization"""
    def rvs(self, a, scale=1, size=1):
        return stats.gamma.rvs(a, scale=scale, size=size)

    def logpdf(self, x, a, scale=1):
        x = tf.cast(x, dtype=tf.float32)
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return (a - 1.0) * tf.log(x) - x/scale - a * tf.log(scale) - lgamma(a)

    def entropy(self, a, scale=1):
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return a + tf.log(scale) + lgamma(a) + \
               tf.mul(1.0 - a, digamma(a))

class Geom:
    def rvs(self, p, size=1):
        return stats.geom.rvs(p, size=size)

    def logpmf(self, x, p):
        x = tf.cast(x, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.mul(x-1, tf.log(1.0-p)) + tf.log(p)

    def entropy(self, p):
        raise NotImplementedError()

class InvGamma:
    """Shape/scale parameterization"""
    def rvs(self, alpha, scale=1, size=1):
        x = stats.invgamma.rvs(alpha, scale=scale, size=size)
        # This is temporary to avoid returning Inf values.
        x[x < 1e-10] = 0.1
        x[x > 1e10] = 1.0
        x[np.logical_not(np.isfinite(x))] = 1.0
        return x

    def logpdf(self, x, a, scale=1):
        x = tf.cast(x, dtype=tf.float32)
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return tf.mul(a, tf.log(scale)) - lgamma(a) + \
               tf.mul(-a-1, tf.log(x)) - tf.truediv(scale, x)

    def entropy(self, a, scale=1):
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return a + tf.log(scale*tf.exp(lgamma(a))) - \
               (1.0 + a) * digamma(a)

class LogNorm:
    def rvs(self, s, size=1):
        return stats.lognorm.rvs(s, size=size)

    def logpdf(self, x, s):
        x = tf.cast(x, dtype=tf.float32)
        s = tf.cast(s, dtype=tf.float32)
        return -0.5*tf.log(2*np.pi) - tf.log(s) - tf.log(x) - \
               0.5*tf.square(tf.log(x) / s)

    def entropy(self, s):
        raise NotImplementedError()

class Multinomial:
    """There is no equivalent version implemented in SciPy."""
    def rvs(self, n, p, size=1):
        return np.random.multinomial(n, p, size=size)

    def logpmf(self, x, n, p):
        """
        Parameters
        ----------
        x : np.array or tf.Tensor
            vector of length K, where x[i] is the number of outcomes
            in the ith bucket, or matrix with column length K
        n : int or tf.Tensor
            number of outcomes equal to sum x[i]
        p : np.array or tf.Tensor
            vector of probabilities summing to 1
        """
        x = tf.cast(x, dtype=tf.float32)
        n = tf.cast(n, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        if len(get_dims(x)) == 1:
            return lgamma(n + 1.0) - \
                   tf.reduce_sum(lgamma(x + 1.0)) + \
                   tf.reduce_sum(tf.mul(x, tf.log(p)))
        else:
            return lgamma(n + 1.0) - \
                   tf.reduce_sum(lgamma(x + 1.0), 1) + \
                   tf.reduce_sum(tf.mul(x, tf.log(p)), 1)

    def entropy(self, n, p):
        # Note that given n and p where p is a probability vector of
        # length k, the entropy requires a sum over all
        # possible configurations of a k-vector which sums to n. It's
        # expensive.
        # http://stackoverflow.com/questions/36435754/generating-a-numpy-array-with-all-combinations-of-numbers-that-sum-to-less-than
        sess = tf.Session()
        n = sess.run(tf.cast(tf.squeeze(n), dtype=tf.int32))
        sess.close()
        p = tf.cast(tf.squeeze(p), dtype=tf.float32)
        if isinstance(n, np.int32):
            k = get_dims(p)[0]
            max_range = np.zeros(k, dtype=np.int32) + n
            x = np.array([i for i in product(*(range(i+1) for i in max_range))
                                 if sum(i)==n])
            logpmf = self.logpmf(x, n, p)
            return tf.reduce_sum(tf.mul(tf.exp(logpmf), logpmf))
        else:
            out = []
            for j in range(n.shape[0]):
                k = get_dims(p)[0]
                max_range = np.zeros(k, dtype=np.int32) + n[j]
                x = np.array([i for i in product(*(range(i+1) for i in max_range))
                                     if sum(i)==n[j]])
                logpmf = self.logpmf(x, n[j], p[j, :])
                out += [tf.reduce_sum(tf.mul(tf.exp(logpmf), logpmf))]

            return tf.pack(out)

class Multivariate_Normal:
    def rvs(self, mean=None, cov=1, size=1):
        return stats.multivariate_normal.rvs(mean, cov, size=size)

    def logpdf(self, x, mean=None, cov=1):
        """
        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        mean : np.array or tf.Tensor, optional
            vector. Defaults to zero mean.
        cov : np.array or tf.Tensor, optional
            vector or matrix. Defaults to identity.
        """
        x = tf.cast(tf.convert_to_tensor(x), dtype=tf.float32)
        x_shape = get_dims(x)
        if len(x_shape) == 1:
            d = x_shape[0]
        else:
            d = x_shape[1]

        if mean is None:
            r = x
        else:
            mean = tf.cast(tf.convert_to_tensor(mean), dtype=tf.float32)
            r = x - mean

        if cov is 1:
            cov_inv = tf.diag(tf.ones([d]))
            det_cov = tf.constant(1.0)
        else:
            cov = tf.cast(tf.convert_to_tensor(cov), dtype=tf.float32)
            if len(cov.get_shape()) == 1: # vector
                cov_inv = tf.diag(1.0 / cov)
                det_cov = tf.reduce_prod(cov)
            else: # matrix
                cov_inv = tf.matrix_inverse(cov)
                det_cov = tf.matrix_determinant(cov)

        lps = -0.5*d*tf.log(2*np.pi) - 0.5*tf.log(det_cov)
        if len(x_shape) == 1:
            r = tf.reshape(r, shape=(d, 1))
            lps -= 0.5 * tf.matmul(tf.matmul(r, cov_inv, transpose_a=True), r)
            return tf.squeeze(lps)
        else:
            # TODO vectorize further
            out = []
            for r_vec in tf.unpack(r):
                r_vec = tf.reshape(r_vec, shape=(d, 1))
                out += [tf.squeeze(lps - 0.5 * tf.matmul(
                                   tf.matmul(r_vec, cov_inv, transpose_a=True),
                                   r_vec))]
            return tf.pack(out)
        """
        # TensorFlow can't reverse-mode autodiff Cholesky
        L = tf.cholesky(cov)
        L_inv = tf.matrix_inverse(L)
        det_cov = tf.pow(tf.matrix_determinant(L), 2)
        inner = dot(L_inv, r)
        out = -0.5*d*tf.log(2*np.pi) - \
              0.5*tf.log(det_cov) - \
              0.5*tf.matmul(inner, inner, transpose_a=True)
        """

    def entropy(self, mean=None, cov=1):
        """
        Note entropy does not depend on the mean.
        This is not vectorized with respect to any arguments.

        Parameters
        ----------
        mean : np.array or tf.Tensor, optional
            vector. Defaults to zero mean.
        cov : np.array or tf.Tensor, optional
            vector or matrix. Defaults to identity.
        """
        if cov is 1:
            d = 1
            det_cov = 1.0
        else:
            cov = tf.cast(tf.convert_to_tensor(cov), dtype=tf.float32)
            d = get_dims(cov)[0]
            if len(cov.get_shape()) == 1:
                det_cov = tf.reduce_prod(cov)
            else:
                det_cov = tf.matrix_determinant(cov)

        return 0.5 * (d + d*tf.log(2*np.pi) + tf.log(det_cov))

class NBinom:
    def rvs(self, n, p, size=1):
        return stats.nbinom.rvs(n, p, size=size)

    def logpmf(self, x, n, p):
        x = tf.cast(x, dtype=tf.float32)
        n = tf.cast(n, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return lgamma(x + n) - lgamma(x + 1.0) - lgamma(n) + \
               tf.mul(n, tf.log(p)) + tf.mul(x, tf.log(1.0-p))

    def entropy(self, n, p):
        raise NotImplementedError()

class Norm:
    def rvs(self, loc=0, scale=1, size=1):
        return stats.norm.rvs(loc, scale, size=size)

    def logpdf(self, x, loc=0, scale=1):
        x = tf.cast(x, dtype=tf.float32)
        loc = tf.cast(loc, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        z = (x - loc) / scale
        return -0.5*tf.log(2*np.pi) - tf.log(scale) - 0.5*tf.square(z)

    def entropy(self, loc=0, scale=1):
        """Note entropy does not depend on the mean."""
        scale = tf.cast(scale, dtype=tf.float32)
        return 0.5 * (1 + tf.log(2*np.pi)) + tf.log(scale)

class Poisson:
    def rvs(self, mu, size=1):
        return stats.poisson.rvs(mu, size=size)

    def logpmf(self, x, mu):
        x = tf.cast(x, dtype=tf.float32)
        mu = tf.cast(mu, dtype=tf.float32)
        return x * tf.log(mu) - mu - lgamma(x + 1.0)

    def entropy(self, mu):
        raise NotImplementedError()

class T:
    def rvs(self, df, loc=0, scale=1, size=1):
        return stats.t.rvs(df, loc=loc, scale=scale, size=size)

    def logpdf(self, x, df, loc=0, scale=1):
        x = tf.cast(x, dtype=tf.float32)
        df = tf.cast(df, dtype=tf.float32)
        loc = tf.cast(loc, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        z = (x - loc) / scale
        return lgamma(0.5 * (df + 1.0)) - lgamma(0.5 * df) - \
               0.5 * (tf.log(np.pi) + tf.log(df)) - tf.log(scale) - \
               0.5 * (df + 1.0) * tf.log(1.0 + (1.0/df) * tf.square(z))

    def entropy(self, df, loc=0, scale=1):
        raise NotImplementedError()

class TruncNorm:
    def rvs(self, a, b, loc=0, scale=1, size=1):
        return stats.truncnorm.rvs(a, b, loc, scale, size=size)

    def logpdf(self, x, a, b, loc=0, scale=1):
        # Note there is no error checking if x is outside domain.
        x = tf.cast(x, dtype=tf.float32)
        # This is slow, as we require use of stats.norm.cdf.
        sess = tf.Session()
        a = sess.run(tf.cast(a, dtype=tf.float32))
        b = sess.run(tf.cast(b, dtype=tf.float32))
        loc = sess.run(tf.cast(loc, dtype=tf.float32))
        scale = sess.run(tf.cast(scale, dtype=tf.float32))
        sess.close()
        return -tf.log(scale) + norm.logpdf(x, loc, scale) - \
               tf.log(tf.cast(stats.norm.cdf((b - loc)/scale) - \
                      stats.norm.cdf((a - loc)/scale),
                      dtype=tf.float32))

    def entropy(self, a, b, loc=0, scale=1):
        raise NotImplementedError()

class Uniform:
    def rvs(self, loc=0, scale=1, size=1):
        return stats.uniform.rvs(loc, scale, size=size)

    def logpdf(self, x, loc=0, scale=1):
        # Note there is no error checking if x is outside domain.
        scale = tf.cast(scale, dtype=tf.float32)
        return tf.squeeze(tf.ones(get_dims(x)) * -tf.log(scale))

    def entropy(self, loc=0, scale=1):
        scale = tf.cast(scale, dtype=tf.float32)
        return tf.log(scale)

bernoulli = Bernoulli()
beta = Beta()
binom = Binom()
chi2 = Chi2()
dirichlet = Dirichlet()
expon = Expon()
gamma = Gamma()
geom = Geom()
invgamma = InvGamma()
lognorm = LogNorm()
multinomial = Multinomial()
multivariate_normal = Multivariate_Normal()
nbinom = NBinom()
norm = Norm()
poisson = Poisson()
t = T()
truncnorm = TruncNorm()
uniform = Uniform()
