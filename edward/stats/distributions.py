import numpy as np
import tensorflow as tf

from edward.util import dot, get_dims, digamma, lbeta, lgamma
from itertools import product
from scipy import stats

class Distribution:
    """Template for all distributions."""
    def rvs(self, size=1):
        """
        Parameters
        ----------
        size : int, list, or tuple, optional
            Number of samples, in a particular shape if specified in a
            list or tuple with more than one element.

        params : float, np.array, or tf.Tensor

        Returns
        -------
        np.ndarray
            np.array of dimension (size x shape), where shape is the
            shape of its parameter argument. For multivariate
            distributions, shape corresponds to one of the parameter
            arguments, e.g., alpha in Dirichlet, p in Multinomial,
            mean in Multivariate_Normal.

        Notes
        -----
        This is written in NumPy/SciPy, as TensorFlow does not support
        many distributions for random number generation. It follows
        SciPy's naming and argument conventions.

        The equivalent method in SciPy is not guaranteed to be
        supported with a batch of parameter inputs, e.g., a vector of
        location parameters in a normal distribution, or a matrix of
        concentration parameters in a Dirichlet. This is.

        This does not follow SciPy's behavior, e.g., the number (or
        shape) of the draws will always be denoted by its outer
        dimension(s).

        params as a 3d or higher tensor is not guaranteed to be
        supported (for either univariate or multivariate
        distribution).

        size as a list or tuple of more than one element is not
        guaranteed to be supported.

        For most distributions, the parameters must be of the same
        shape and type, e.g., n and p in Binomial must be np.arrays()
        of same shape or both floats. For some, they may differ by one
        dimension, e.g., n and p in Multinomial can be float and
        np.array(), or both np.arrays, and n always has one less
        dimension.
        """
        raise NotImplementedError()

    def logpmf(self, x):
        """
        Parameters
        ---------
        x : float, np.array, or tf.Tensor
            If univariate distribution, can be a scalar or tensor.
            If multivariate distribution, can be a tensor; the outer
            dimension carries the multivariate dimension.

        params : float, np.array, or tf.Tensor
            scalar unless documented otherwise

        Returns
        -------
        tf.Tensor
            If univariate distribution, returns a tensor of same
            shape as input.
            If multivariate distribution, returns a tensor of
            shape[-1] from input: the outer dimension representing the
            multivariate dimension is collapsed.

        Notes
        -----
        x as a 3d or higher tensor is not guaranteed to be supported
        (for either univariate or multivariate distribution).
        """
        raise NotImplementedError()

    def entropy(self):
        """
        Parameters
        ---------
        params : float, np.array, or tf.Tensor
            If univariate distribution, can be a scalar, vector, or
            matrix.
            If multivariate distribution, can be a vector or matrix.

        Returns
        -------
        tf.Tensor
            If univariate distribution, returns a tensor of same
            shape as input.
            If multivariate distribution, returns a tensor of
            shape[-1] from input: the outer dimension representing the
            multivariate dimension is collapsed.

        Notes
        -----
        The equivalent method in SciPy is not guaranteed to be
        supported for vector inputs for univariate distributions and
        matrix inputs for multivariate distributions. This is.

        params as a 3d or higher tensor is not guaranteed to be
        supported (for either univariate or multivariate
        distribution).
        """
        raise NotImplementedError()

class Bernoulli:
    def rvs(self, p, size=1):
        """
        Examples
        --------
        >>> x = bernoulli.rvs(p=0.5, size=1)
        >>> print(x.shape)
        (1, 1)
        >>> x = bernoulli.rvs(p=np.array([0.5]), size=1)
        >>> print(x.shape)
        (1, 1)
        >>> x = bernoulli.rvs(p=np.array([0.5, 0.2]), size=1)
        >>> print(x.shape)
        (1, 2)
        """
        if not isinstance(p, np.ndarray):
            p = np.asarray(p)
        if len(p.shape) == 0:
            return stats.bernoulli.rvs(p, size=size)

        x = []
        for pidx in np.nditer(p):
            x += [stats.bernoulli.rvs(pidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

    def logpmf(self, x, p):
        x = tf.cast(x, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.mul(x, tf.log(p)) + tf.mul(1.0 - x, tf.log(1.0-p))

    def entropy(self, p):
        p = tf.cast(p, dtype=tf.float32)
        return -tf.mul(p, tf.log(p)) - tf.mul(1.0 - p, tf.log(1.0-p))

class Beta:
    def rvs(self, a, b, size=1):
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        if not isinstance(b, np.ndarray):
            b = np.asarray(b)
        if len(a.shape) == 0:
            return stats.beta.rvs(a, b, size=size)

        x = []
        for aidx,bidx in zip(np.nditer(a), np.nditer(b)):
            x += [stats.beta.rvs(aidx, bidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if not isinstance(n, np.ndarray):
            n = np.asarray(n)
        if not isinstance(p, np.ndarray):
            p = np.asarray(p)
        if len(n.shape) == 0:
            return stats.binom.rvs(n, p, size=size)

        x = []
        for nidx,pidx in zip(np.nditer(n), np.nditer(p)):
            x += [stats.binom.rvs(nidx, pidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if not isinstance(df, np.ndarray):
            df = np.asarray(df)
        if len(df.shape) == 0:
            return stats.chi2.rvs(df, size=size)

        x = []
        for dfidx in np.nditer(df):
            x += [stats.chi2.rvs(dfidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

    def logpdf(self, x, df):
        x = tf.cast(x, dtype=tf.float32)
        df = tf.cast(df, dtype=tf.float32)
        return tf.mul(0.5*df - 1, tf.log(x)) - 0.5*x - \
               tf.mul(0.5*df, tf.log(2.0)) - lgamma(0.5*df)

    def entropy(self, df):
        raise NotImplementedError()

class Dirichlet:
    def rvs(self, alpha, size=1):
        if len(alpha.shape) == 1:
            # stats.dirichlet.rvs defaults to (size x alpha.shape)
            return stats.dirichlet.rvs(alpha, size=size)

        x = []
        # This doesn't work for non-matrix parameters.
        for alpharow in alpha:
            x += [stats.dirichlet.rvs(alpharow, size=size)]

        # This only works for rank 3 tensor.
        x = np.rollaxis(np.asarray(x), 1)
        return x

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
        if not isinstance(scale, np.ndarray):
            scale = np.asarray(scale)
        if len(scale.shape) == 0:
            return stats.expon.rvs(scale=scale, size=size)

        x = []
        for scaleidx in np.nditer(scale):
            x += [stats.expon.rvs(scale=scaleidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

    def logpdf(self, x, scale=1):
        x = tf.cast(x, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return - x/scale - tf.log(scale)

    def entropy(self, scale=1):
        raise NotImplementedError()

class Gamma:
    """Shape/scale parameterization"""
    def rvs(self, a, scale=1, size=1):
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        if not isinstance(scale, np.ndarray):
            scale = np.asarray(scale)
        if len(a.shape) == 0:
            return stats.gamma.rvs(a, scale=scale, size=size)

        x = []
        for aidx,scaleidx in zip(np.nditer(a), np.nditer(scale)):
            x += [stats.gamma.rvs(aidx, scale=scaleidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if not isinstance(p, np.ndarray):
            p = np.asarray(p)
        if len(p.shape) == 0:
            return stats.geom.rvs(p, size=size)

        x = []
        for pidx in np.nditer(p):
            x += [stats.geom.rvs(pidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

    def logpmf(self, x, p):
        x = tf.cast(x, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.mul(x-1, tf.log(1.0-p)) + tf.log(p)

    def entropy(self, p):
        raise NotImplementedError()

class InvGamma:
    """Shape/scale parameterization"""
    def rvs(self, a, scale=1, size=1):
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        if not isinstance(scale, np.ndarray):
            scale = np.asarray(scale)
        if len(a.shape) == 0:
            return stats.invgamma.rvs(a, scale=scale, size=size)

        x = []
        for aidx,scaleidx in zip(np.nditer(a), np.nditer(scale)):
            x += [stats.invgamma.rvs(aidx, scale=scaleidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()

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
        if not isinstance(s, np.ndarray):
            s = np.asarray(s)
        if len(s.shape) == 0:
            return stats.lognorm.rvs(s, size=size)

        x = []
        for sidx in np.nditer(s):
            x += [stats.lognorm.rvs(sidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if len(p.shape) == 1:
            # np.random.multinomial defaults to (size x p.shape)
            return np.random.multinomial(n, p, size=size)

        if not isinstance(n, np.ndarray):
            n = np.asarray(n)

        x = []
        # This doesn't work for non-matrix parameters.
        for nidx,prow in zip(n, p):
            x += [np.random.multinomial(nidx, prow, size=size)]

        # This only works for rank 3 tensor.
        x = np.rollaxis(np.asarray(x), 1)
        return x

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
        if len(mean.shape) == 1:
            x = stats.multivariate_normal.rvs(mean, cov, size=size)
            # stats.multivariate_normal.rvs returns (size x shape) if
            # size > 1, and shape if size == 1.
            if size == 1:
                return np.expand_dims(x, axis=0)
            else:
                return x

        x = []
        # This doesn't work for non-matrix parameters.
        for meanrow,covmat in zip(mean, cov):
            x += [stats.multivariate_normal.rvs(meanrow, covmat, size=size)]

        # This only works for rank 3 tensor.
        x = np.rollaxis(np.asarray(x), 1)
        return x

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
        if not isinstance(n, np.ndarray):
            n = np.asarray(n)
        if not isinstance(p, np.ndarray):
            p = np.asarray(p)
        if len(n.shape) == 0:
            return stats.nbinom.rvs(n, p, size=size)

        x = []
        for nidx,pidx in zip(np.nditer(n), np.nditer(p)):
            x += [stats.nbinom.rvs(nidx, pidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if not isinstance(loc, np.ndarray):
            loc = np.asarray(loc)
        if not isinstance(scale, np.ndarray):
            scale = np.asarray(scale)
        if len(loc.shape) == 0:
            return stats.norm.rvs(loc, scale, size=size)

        x = []
        for locidx,scaleidx in zip(np.nditer(loc), np.nditer(scale)):
            x += [stats.norm.rvs(locidx, scaleidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if not isinstance(mu, np.ndarray):
            mu = np.asarray(mu)
        if len(mu.shape) == 0:
            return stats.poisson.rvs(mu, size=size)

        x = []
        for muidx in np.nditer(mu):
            x += [stats.poisson.rvs(muidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

    def logpmf(self, x, mu):
        x = tf.cast(x, dtype=tf.float32)
        mu = tf.cast(mu, dtype=tf.float32)
        return x * tf.log(mu) - mu - lgamma(x + 1.0)

    def entropy(self, mu):
        raise NotImplementedError()

class T:
    def rvs(self, df, loc=0, scale=1, size=1):
        if not isinstance(df, np.ndarray):
            df = np.asarray(df)
        if not isinstance(loc, np.ndarray):
            loc = np.asarray(loc)
        if not isinstance(scale, np.ndarray):
            scale = np.asarray(scale)
        if len(df.shape) == 0:
            return stats.t.rvs(df, loc=loc, scale=scale, size=size)

        x = []
        for dfidx,locidx,scaleidx in zip(np.nditer(df), np.nditer(loc), np.nditer(scale)):
            x += [stats.t.rvs(dfidx, loc=locidx, scale=scaleidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        if not isinstance(b, np.ndarray):
            b = np.asarray(b)
        if not isinstance(loc, np.ndarray):
            loc = np.asarray(loc)
        if not isinstance(scale, np.ndarray):
            scale = np.asarray(scale)
        if len(a.shape) == 0:
            return stats.truncnorm.rvs(a, b, loc, scale, size=size)

        x = []
        for aidx,bidx,locidx,scaleidx in zip(np.nditer(a), np.nditer(b), np.nditer(loc), np.nditer(scale)):
            x += [stats.truncnorm.rvs(aidx, bidx, locidx, scaleidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
        if not isinstance(loc, np.ndarray):
            loc = np.asarray(loc)
        if not isinstance(scale, np.ndarray):
            scale = np.asarray(scale)
        if len(loc.shape) == 0:
            return stats.uniform.rvs(loc, scale, size=size)

        x = []
        for locidx,scaleidx in zip(np.nditer(loc), np.nditer(scale)):
            x += [stats.uniform.rvs(locidx, scaleidx, size=size)]

        # Note this doesn't work for multi-dimensional sizes.
        x = np.asarray(x).transpose()
        return x

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
