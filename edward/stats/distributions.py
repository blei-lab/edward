import numpy as np
import tensorflow as tf

from edward.util import dot, get_dims
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
    """Bernoulli distribution
    """    
    def rvs(self, p, size=1):
        """Random variable generator

        Parameters
        ----------
        p : float
            constrained to :math:`p\in(0,1)`
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        
        """        
        return stats.bernoulli.rvs(p, size=size)

    def logpmf(self, x, p):
        """Logarithm of probability mass function

        Parameters
        ----------
        x : np.array or tf.Tensor
            If univariate distribution, can be a scalar, vector, or matrix.
            If multivariate distribution, can be a vector or matrix.
        p : float
            constrained to :math:`p\in(0,1)`
        
        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.
        """
        x = tf.cast(x, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.mul(x, tf.log(p)) + tf.mul(1.0 - x, tf.log(1.0-p))

    def entropy(self, p):
        """Entropy of probability distribution

        Parameters
        ----------
        p : float
            constrained to :math:`p\in(0,1)`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar or vector
            corresponding to the size of input. For multivariate
            distributions, returns a scalar if vector input and vector
            if matrix input, where each element in the vector
            evaluates a row in the matrix.
        """        
        p = tf.cast(p, dtype=tf.float32)
        return -tf.mul(p, tf.log(p)) - tf.mul(1.0 - p, tf.log(1.0-p))

class Beta:
    """Beta distribution
    """    
    def rvs(self, a, b, size=1):
        """Random variable generator

        Parameters
        ----------
        a : float
            constrained to :math:`a > 0`
        b : float
            constrained to :math:`b > 0`            
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """               
        return stats.beta.rvs(a, b, size=size)

    def logpdf(self, x, a, b):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            If univariate distribution, can be a scalar, vector, or matrix.
            If multivariate distribution, can be a vector or matrix.
        a : float
            constrained to :math:`a > 0`
        b : float
            constrained to :math:`b > 0` 
        
        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.
        """        
        x = tf.cast(x, dtype=tf.float32)
        a = tf.cast(tf.squeeze(a), dtype=tf.float32)
        b = tf.cast(tf.squeeze(b), dtype=tf.float32)
        return (a-1) * tf.log(x) + (b-1) * tf.log(1-x) - tf.lbeta(tf.pack([a, b]))

    def entropy(self, a, b):
        """Entropy of probability distribution

        Parameters
        ----------
        a : float
            constrained to :math:`a > 0`
        b : float
            constrained to :math:`b > 0` 

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar or vector
            corresponding to the size of input. For multivariate
            distributions, returns a scalar if vector input and vector
            if matrix input, where each element in the vector
            evaluates a row in the matrix.
        """          
        a = tf.cast(tf.squeeze(a), dtype=tf.float32)
        b = tf.cast(tf.squeeze(b), dtype=tf.float32)
        if len(a.get_shape()) == 0:
            return tf.lbeta(tf.pack([a, b])) - \
                   tf.mul(a - 1.0, tf.digamma(a)) - \
                   tf.mul(b - 1.0, tf.digamma(b)) + \
                   tf.mul(a + b - 2.0, tf.digamma(a+b))
        else:
            return tf.lbeta(tf.concat(1,
                         [tf.expand_dims(a, 1), tf.expand_dims(b, 1)])) - \
                   tf.mul(a - 1.0, tf.digamma(a)) - \
                   tf.mul(b - 1.0, tf.digamma(b)) + \
                   tf.mul(a + b - 2.0, tf.digamma(a+b))

class Binom:
    """Binomial distribution
    """    
    def rvs(self, n, p, size=1):
        """Random variable generator

        Parameters
        ----------
        n : int
            constrained to :math:`n > 0`
        p : float
            constrained to :math:`p\in(0,1)`      
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """            
        return stats.binom.rvs(n, p, size=size)

    def logpmf(self, x, n, p):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            If univariate distribution, can be a scalar, vector, or matrix.
            If multivariate distribution, can be a vector or matrix.
        n : int
            constrained to :math:`n > 0`
        p : float
            constrained to :math:`p\in(0,1)`      
        
        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.
        """            
        x = tf.cast(x, dtype=tf.float32)
        n = tf.cast(n, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.lgamma(n + 1.0) - tf.lgamma(x + 1.0) - tf.lgamma(n - x + 1.0) + \
               tf.mul(x, tf.log(p)) + tf.mul(n - x, tf.log(1.0-p))

    def entropy(self, n, p):
        """
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

class Chi2:
    """:math:`\chi^2` distribution
    """    
    def rvs(self, df, size=1):
        """Random variable generator

        Parameters
        ----------
        df : float
            constrained to :math:`df > 0`      
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """          
        return stats.chi2.rvs(df, size=size)

    def logpdf(self, x, df):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            If univariate distribution, can be a scalar, vector, or matrix.
            If multivariate distribution, can be a vector or matrix.
        df : float
            constrained to :math:`df > 0`
          
        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.
        """          
        x = tf.cast(x, dtype=tf.float32)
        df = tf.cast(df, dtype=tf.float32)
        return tf.mul(0.5*df - 1, tf.log(x)) - 0.5*x - \
               tf.mul(0.5*df, tf.log(2.0)) - tf.lgamma(0.5*df)

    def entropy(self, df):
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class Dirichlet:
    """Dirichlet distribution
    """    
    def rvs(self, alpha, size=1):
        """Random variable generator

        Parameters
        ----------
        alpha : np.array or tf.Tensor
            each :math:`\alpha` constrained to :math:`\alpha_i > 0`      
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """          
        return stats.dirichlet.rvs(alpha, size=size)

    def logpdf(self, x, alpha):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        alpha : np.array or tf.Tensor
            each :math:`\alpha` constrained to :math:`\alpha_i > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.
        """
        x = tf.cast(x, dtype=tf.float32)
        alpha = tf.cast(tf.convert_to_tensor(alpha), dtype=tf.float32)
        if len(get_dims(x)) == 1:
            return -tf.lbeta(alpha) + tf.reduce_sum(tf.mul(alpha-1, tf.log(x)))
        else:
            return -tf.lbeta(alpha) + tf.reduce_sum(tf.mul(alpha-1, tf.log(x)), 1)

    def entropy(self, alpha):
        """Entropy of probability distribution

        Parameters
        ----------
        alpha : np.array or tf.Tensor
            each :math:`\alpha` constrained to :math:`\alpha_i > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar or vector
            corresponding to the size of input. For multivariate
            distributions, returns a scalar if vector input and vector
            if matrix input, where each element in the vector
            evaluates a row in the matrix.            
        """
        alpha = tf.cast(tf.convert_to_tensor(alpha), dtype=tf.float32)
        if len(get_dims(alpha)) == 1:
            K = get_dims(alpha)[0]
            a = tf.reduce_sum(alpha)
            return tf.lbeta(alpha) + \
                   tf.mul(a - K, tf.digamma(a)) - \
                   tf.reduce_sum(tf.mul(alpha-1, tf.digamma(alpha)))
        else:
            K = get_dims(alpha)[1]
            a = tf.reduce_sum(alpha, 1)
            return tf.lbeta(alpha) + \
                   tf.mul(a - K, tf.digamma(a)) - \
                   tf.reduce_sum(tf.mul(alpha-1, tf.digamma(alpha)), 1)

class Expon:
    """Exponential distribution
    """
    def rvs(self, scale=1, size=1):
        """Random variable generator

        Parameters
        ----------
        scale : float
            constrained to :math:`scale > 0`    
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """          
        return stats.expon.rvs(scale=scale, size=size)

    def logpdf(self, x, scale=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        scale : float
            constrained to :math:`scale > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.
        """        
        x = tf.cast(x, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return - x/scale - tf.log(scale)

    def entropy(self, scale=1):
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class Gamma:
    """Gamma distribution

    Shape/scale parameterization (typically denoted: :math:`(k, \\theta)`)
    """
    def rvs(self, a, scale=1, size=1):
        """Random variable generator

        Parameters
        ----------
        a : float
            **shape** parameter: constrained to :math:`a > 0`    
        scale : float
            **scale** parameter: constrained to :math:`scale > 0`  
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """           
        return stats.gamma.rvs(a, scale=scale, size=size)

    def logpdf(self, x, a, scale=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        a : float
            **shape** parameter: constrained to :math:`a > 0`    
        scale : float
            **scale** parameter: constrained to :math:`scale > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
        """           
        x = tf.cast(x, dtype=tf.float32)
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return (a - 1.0) * tf.log(x) - x/scale - a * tf.log(scale) - tf.lgamma(a)

    def entropy(self, a, scale=1):
        """Entropy of probability distribution

        Parameters
        ----------
        a : float
            **shape** parameter: constrained to :math:`a > 0`    
        scale : float
            **scale** parameter: constrained to :math:`scale > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar or vector
            corresponding to the size of input. For multivariate
            distributions, returns a scalar if vector input and vector
            if matrix input, where each element in the vector
            evaluates a row in the matrix.            
        """        
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return a + tf.log(scale) + tf.lgamma(a) + \
               tf.mul(1.0 - a, tf.digamma(a))

class Geom:
    """Geometric distribution
    """
    def rvs(self, p, size=1):
        """Random variable generator

        Parameters
        ----------
        p : float
            constrained to :math:`p\in(0,1)` 
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """             
        return stats.geom.rvs(p, size=size)

    def logpmf(self, x, p):
        """Logarithm of probability mass function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        p : float
            constrained to :math:`p\in(0,1)` 

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
        """         
        x = tf.cast(x, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.mul(x-1, tf.log(1.0-p)) + tf.log(p)

    def entropy(self, p):
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class InvGamma:
    """Inverse Gamma distribution

    Shape/scale parameterization (typically denoted: :math:`(k, \\theta)`)
    """
    def rvs(self, a, scale=1, size=1):
        """Random variable generator

        Parameters
        ----------
        a : float
            **shape** parameter: constrained to :math:`a > 0`    
        scale : float
            **scale** parameter: constrained to :math:`scale > 0`  
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """          
        x = stats.invgamma.rvs(a, scale=scale, size=size)
        # This is temporary to avoid returning Inf values.
        x[x < 1e-10] = 0.1
        x[x > 1e10] = 1.0
        x[np.logical_not(np.isfinite(x))] = 1.0
        return x

    def logpdf(self, x, a, scale=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        a : float
            **shape** parameter: constrained to :math:`a > 0`    
        scale : float
            **scale** parameter: constrained to :math:`scale > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
        """            
        x = tf.cast(x, dtype=tf.float32)
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return tf.mul(a, tf.log(scale)) - tf.lgamma(a) + \
               tf.mul(-a-1, tf.log(x)) - tf.truediv(scale, x)

    def entropy(self, a, scale=1):
        """Entropy of probability distribution

        Parameters
        ----------
        a : float
            **shape** parameter: constrained to :math:`a > 0`    
        scale : float
            **scale** parameter: constrained to :math:`scale > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar or vector
            corresponding to the size of input. For multivariate
            distributions, returns a scalar if vector input and vector
            if matrix input, where each element in the vector
            evaluates a row in the matrix.            
        """          
        a = tf.cast(a, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        return a + tf.log(scale*tf.exp(tf.lgamma(a))) - \
               (1.0 + a) * tf.digamma(a)

class LogNorm:
    """LogNormal distribution
    """
    def rvs(self, s, size=1):
        """Random variable generator

        Parameters
        ---------- 
        s : float
            constrained to :math:`s > 0`  
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """           
        return stats.lognorm.rvs(s, size=size)

    def logpdf(self, x, s):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        s : float
            constrained to :math:`s > 0`  

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
        """           
        x = tf.cast(x, dtype=tf.float32)
        s = tf.cast(s, dtype=tf.float32)
        return -0.5*tf.log(2*np.pi) - tf.log(s) - tf.log(x) - \
               0.5*tf.square(tf.log(x) / s)

    def entropy(self, s):
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class Multinomial:
    """Multinomial distribution

    Note: there is no equivalent version implemented in SciPy.
    """
    def rvs(self, n, p, size=1):
        """Random variable generator

        Parameters
        ---------- 
        n : int
            constrained to :math:`n > 0`
        p : np.array
            constrained to :math:`\sum_i p_k = 1`  
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """          
        return np.random.multinomial(n, p, size=size)

    def logpmf(self, x, n, p):
        """Logarithm of probability mass function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector of length K, where x[i] is the number of outcomes
            in the ith bucket, or matrix with column length K
        n : int or tf.Tensor
            number of outcomes equal to sum x[i]
        p : np.array or tf.Tensor
            vector of probabilities summing to 1

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
        """  
        x = tf.cast(x, dtype=tf.float32)
        n = tf.cast(n, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        if len(get_dims(x)) == 1:
            return tf.lgamma(n + 1.0) - \
                   tf.reduce_sum(tf.lgamma(x + 1.0)) + \
                   tf.reduce_sum(tf.mul(x, tf.log(p)))
        else:
            return tf.lgamma(n + 1.0) - \
                   tf.reduce_sum(tf.lgamma(x + 1.0), 1) + \
                   tf.reduce_sum(tf.mul(x, tf.log(p)), 1)

    def entropy(self, n, p):
        """TODO
        """
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
    """Multivariate Normal distribution
    """
    def rvs(self, mean=None, cov=1, size=1):
        """Random variable generator

        Parameters
        ---------- 
        mean : np.array or tf.Tensor, optional
            vector. Defaults to zero mean.
        cov : np.array or tf.Tensor, optional
            vector or matrix. Defaults to identity matrix.
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """  
        return stats.multivariate_normal.rvs(mean, cov, size=size)

    def logpdf(self, x, mean=None, cov=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        mean : np.array or tf.Tensor, optional
            vector. Defaults to zero mean.
        cov : np.array or tf.Tensor, optional
            vector or matrix. Defaults to identity matrix.

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.              
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
        """Entropy of probability distribution

        This is not vectorized with respect to any arguments.

        Parameters
        ----------
        mean : np.array or tf.Tensor, optional
            vector. Defaults to zero mean.
        cov : np.array or tf.Tensor, optional
            vector or matrix. Defaults to identity matrix.

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
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
    """Negative binomial distribution
    """
    def rvs(self, n, p, size=1):
        """Random variable generator

        Parameters
        ---------- 
        n : int
            constrained to :math:`n > 0`
        p : float
            constrained to :math:`p\in(0,1)` 
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """          
        return stats.nbinom.rvs(n, p, size=size)

    def logpmf(self, x, n, p):
        """Logarithm of probability mass function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        n : int
            constrained to :math:`n > 0`
        p : float
            constrained to :math:`p\in(0,1)` 

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.              
        """        
        x = tf.cast(x, dtype=tf.float32)
        n = tf.cast(n, dtype=tf.float32)
        p = tf.cast(p, dtype=tf.float32)
        return tf.lgamma(x + n) - tf.lgamma(x + 1.0) - tf.lgamma(n) + \
               tf.mul(n, tf.log(p)) + tf.mul(x, tf.log(1.0-p))

    def entropy(self, n, p):
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class Norm:
    """Normal (Gaussian) distribution
    """
    def rvs(self, loc=0, scale=1, size=1):
        """Random variable generator

        Parameters
        ---------- 
        loc : float
            mean
        scale : float
            standard deviation, constrained to :math:`scale > 0`
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """            
        return stats.norm.rvs(loc, scale, size=size)

    def logpdf(self, x, loc=0, scale=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        loc : float
            mean
        scale : float
            standard deviation, constrained to :math:`scale > 0`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.              
        """         
        x = tf.cast(x, dtype=tf.float32)
        loc = tf.cast(loc, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        z = (x - loc) / scale
        return -0.5*tf.log(2*np.pi) - tf.log(scale) - 0.5*tf.square(z)

    def entropy(self, loc=0, scale=1):
        """Entropy of probability distribution

        Parameters
        ----------
        loc : float
            mean
        scale : float
            standard deviation, constrained to :math:`scale > 0`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
        """
        scale = tf.cast(scale, dtype=tf.float32)
        return 0.5 * (1 + tf.log(2*np.pi)) + tf.log(scale)

class Poisson:
    """Poisson distribution
    """
    def rvs(self, mu, size=1):
        """Random variable generator

        Parameters
        ---------- 
        mu : float
            parameter, constrained to :math:`mu > 0`
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """  
        return stats.poisson.rvs(mu, size=size)

    def logpmf(self, x, mu):
        """Logarithm of probability mass function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        mu : float
            parameter, constrained to :math:`mu > 0`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.              
        """          
        x = tf.cast(x, dtype=tf.float32)
        mu = tf.cast(mu, dtype=tf.float32)
        return x * tf.log(mu) - mu - tf.lgamma(x + 1.0)

    def entropy(self, mu):
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class T:
    """Student T distribution
    """
    def rvs(self, df, loc=0, scale=1, size=1):
        """Random variable generator

        Parameters
        ---------- 
        df : float
            constrained to :math:`df > 0`  
        loc : float
            mean
        scale : float
            standard deviation, constrained to :math:`scale > 0`
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """                
        return stats.t.rvs(df, loc=loc, scale=scale, size=size)

    def logpdf(self, x, df, loc=0, scale=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        df : float
            constrained to :math:`df > 0`  
        loc : float
            mean
        scale : float
            standard deviation, constrained to :math:`scale > 0`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.              
        """         
        x = tf.cast(x, dtype=tf.float32)
        df = tf.cast(df, dtype=tf.float32)
        loc = tf.cast(loc, dtype=tf.float32)
        scale = tf.cast(scale, dtype=tf.float32)
        z = (x - loc) / scale
        return tf.lgamma(0.5 * (df + 1.0)) - tf.lgamma(0.5 * df) - \
               0.5 * (tf.log(np.pi) + tf.log(df)) - tf.log(scale) - \
               0.5 * (df + 1.0) * tf.log(1.0 + (1.0/df) * tf.square(z))

    def entropy(self, df, loc=0, scale=1):
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class TruncNorm:
    """Truncated Normal (Gaussian) distribution
    """
    def rvs(self, a, b, loc=0, scale=1, size=1):
        """Random variable generator

        Parameters
        ---------- 
        a : float
            left boundary, with respect to the standard normal
        b : float
            right boundary, with respect to the standard normal
        loc : float
            mean
        scale : float
            standard deviation, constrained to :math:`scale > 0`
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """           
        return stats.truncnorm.rvs(a, b, loc, scale, size=size)

    def logpdf(self, x, a, b, loc=0, scale=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        a : float
            left boundary, with respect to the standard normal
        b : float
            right boundary, with respect to the standard normal
        loc : float
            mean
        scale : float
            standard deviation, constrained to :math:`scale > 0`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.              
        """         
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
        """
        Raises
        ------
        NotImplementedError
        """        
        raise NotImplementedError()

class Uniform:
    """Uniform distribution (continous)

    This distribution is constant between ``loc`` and ``loc + scale``
    """
    def rvs(self, loc=0, scale=1, size=1):
        """Random variable generator

        Parameters
        ---------- 
        loc : float
            left boundary
        scale : float
            width of distribution, constrained to :math:`scale > 0`
        size : int
            number of random variable samples to return

        Returns
        -------
        np.ndarray
            size-dimensional vector; scalar if size=1    
        """         
        return stats.uniform.rvs(loc, scale, size=size)

    def logpdf(self, x, loc=0, scale=1):
        """Logarithm of probability density function

        Parameters
        ----------
        x : np.array or tf.Tensor
            vector or matrix
        loc : float
            left boundary
        scale : float
            width of distribution, constrained to :math:`scale > 0`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.              
        """          
        # Note there is no error checking if x is outside domain.
        scale = tf.cast(scale, dtype=tf.float32)
        return tf.squeeze(tf.ones(get_dims(x)) * -tf.log(scale))

    def entropy(self, loc=0, scale=1):
        """Entropy of probability distribution

        Parameters
        ----------
        loc : float
            left boundary
        scale : float
            width of distribution, constrained to :math:`scale > 0`

        Returns
        -------
        tf.Tensor
            For univariate distributions, returns a scalar, vector, or
            matrix corresponding to the size of input. For
            multivariate distributions, returns a scalar if vector
            input and vector if matrix input, where each element in
            the vector evaluates a row in the matrix.  
        """
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
