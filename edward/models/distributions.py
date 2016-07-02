from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import bernoulli, beta, norm, dirichlet, invgamma, multinomial
from edward.util import cumprod, get_dims, get_session, to_simplex

class Distribution:
    """Base class for Edward distributions.

    ``p(x | params) = prod_{idx in shape} p(x[idx] | params[idx])``

    where ``shape`` is the shape of ``x`` and ``params[idx]`` denote the
    parameters of each random variable ``x[idx]``.

    Attributes
    ----------
    shape : tuple
        shape of random variable(s); see below
    num_vars : int
        the number of variables; equals the product of ``shape`` 
    num_params : int
        the number of parameters
    sample_tensor : bool
        ``True`` if sampling returns a ``tf.Tensor``; see :func:`sample`
    is_multivariate : bool
        ``True`` if ``Distribution`` is multivariate
    """
    def __init__(self, shape=1):
        """Initialize.

        Parameters
        ----------
        shape : int, list, or tuple, optional
            Shape of random variable(s). If ``is_multivariate=True``, then the
            inner-most (right-most) dimension indicates the dimension of
            the multivariate random variable. Otherwise, all dimensions
            are conceptually the same.
        """
        get_session()
        if isinstance(shape, int):
            shape = (shape, )
        elif isinstance(shape, list):
            shape = tuple(shape)

        self.shape = shape
        self.num_vars = np.prod(self.shape)
        self.num_params = None
        self.sample_tensor = False
        self.is_multivariate = False

    def sample_noise(self, size=1):
        """Sample from a standard parameterization of ``Distribution``.

        For example, if ``epsilon = sample_noise()``, then

        .. math::
            \epsilon \sim s(\epsilon),
    
        where :math:`s` is a standard parameterization of ``Distribution`` 
        that has no parameters.

        Passing :math:`\epsilon` into ``reparam`` then transforms the draw
        according to some parameters.

        Parameters
        ----------
        size : int, optional
            Number of samples to return.

        Returns
        -------
        tf.Tensor
            A ``[size x shape]`` array of type ``tf.float32``, where each
            slice along the first dimension is a sample from ``s(epsilon)``.

        Raises
        ------
        NotImplementedError            
        """
        raise NotImplementedError()

    def reparam(self, eps):
        """Reparameterizes a sample from a standard version of ``Distribution``.

        For example, if ``epsilon = sample_noise()``, then calling 
        ``x = reparam(epsilon)`` reparameterizes ``epsilon`` such that

        .. math::
            x \sim p(x | \\text{params})

        where ``params`` are the parameters that govern this 
        ``Distribution`` object.

        Parameters
        ----------
        eps : tf.Tensor
            A draw from ``sample_noise`` of this ``Distribution`` object.

        Returns
        -------
        tf.Tensor
            A ``[size x shape]`` array of type ``tf.float32``, where each
            slice along the first dimension is a sample from 
            ``p(x | params)``.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def sample(self, size=1):
        """Sample from ``Distribution``.

        .. math::
            x \sim p(x | \\text{params})

        Defaults to sampling noise via ``sample_noise`` and reparameterizing via
        ``reparam``. Otherwise expects a derived class to implement its own 
        method.

        Parameters
        ----------
        size : int, optional
            Number of samples to return.        

        Returns
        -------
        tf.Tensor or np.ndarray
            A (size x shape) array of type tf.float32, where each
            slice along the first dimension is a sample from p. If
            the flag sample_tensor is true, the return object is a
            TensorFlow tensor. Otherwise the return object is a
            realization of a TensorFlow tensor, i.e., NumPy array. The
            latter is required when we require NumPy/SciPy in order to
            sample from distributions.
        """
        return self.reparam(self.sample_noise(size))

    def log_prob(self, xs):
        """Evaluate log probability.

            ``log p(xs | params)``

        Parameters
        ----------
        xs : tf.Tensor or np.ndarray
            n_minibatch x self.shape

        Returns
        -------
        tf.Tensor
            A vector for each log density evaluation,

            .. code-block:: none

                [ sum_{idx in shape} log p(xs[1, idx] | params[idx]), 
                ...,
                sum_{idx in shape} log p(xs[n_minibatch, idx] | params[idx]) ]
        """
        if isinstance(xs, tf.Tensor):
            shape = get_dims(xs)
        else: # NumPy array
            shape = xs.shape

        # Loop over each random variable.
        # If univariate distribution, this is over all indices; if
        # multivariate distribution, this is over all but the last
        # index.
        n_minibatch = shape[0]
        log_prob = tf.zeros([n_minibatch], dtype=tf.float32)
        if len(self.shape) == 1:
            if self.is_multivariate:
                idx = ()
                log_prob += self.log_prob_idx(idx, xs)
            else:
                for i in range(self.shape[0]):
                    idx = (i, )
                    log_prob += self.log_prob_idx(idx, xs)

        elif len(self.shape) == 2:
            if self.is_multivariate:
                for i in range(self.shape[0]):
                    idx = (i, )
                    log_prob += self.log_prob_idx(idx, xs)

            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        idx = (i, j, )
                        log_prob += self.log_prob_idx(idx, xs)

        elif len(self.shape) == 3:
            if self.is_multivariate:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        idx = (i, j, )
                        log_prob += self.log_prob_idx(idx, xs)

            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        for k in range(self.shape[2]):
                            idx = (i, j, k, )
                            log_prob += self.log_prob_idx(idx, xs)

        else: # len(self.shape) >= 4
            # There should be a generic recursive solution.
            raise NotImplementedError()

        return log_prob

    def log_prob_idx(self, idx, xs):
        """Log probability (single index)

            ``log p(xs[:, idx] | params[idx])``

        Parameters
        ----------
        idx : tuple
            Index of the random variable to take the log density of.
            If univariate distribution, idx is of length
            len(self.shape). If multivariate distribution, idx is of
            length len(self.shape[:-1]); note if len(self.shape) is 1
            for multivariate, then idx must be an empty tuple.
        xs : tf.Tensor or np.ndarray
            of size ``[n_minibatch x self.shape]``

        Returns
        -------
        tf.Tensor
            A vector

            .. code-block:: none

                [ log p(xs[1, idx] | params[idx]), 
                ..., 
                log p(xs[S, idx] | params[idx]) ]

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()

    def entropy(self):
        """Entropy.

        .. code-block:: none

            H(p(x| params))
            = E_{p(x | params)} [ - log p(x | params) ]
            = sum_{idx in shape} E_{p(x[idx] | params[idx])} [ - log p(x[idx] | params[idx]) ]

        Returns
        -------
        tf.Tensor
            scalar
        """
        raise NotImplementedError()

class Bernoulli(Distribution):
    """Bernoulli 

    See :class:`edward.stats.distributions.Bernoulli`
        
    """
    def __init__(self, shape=1, p=None):
        Distribution.__init__(self, shape)
        self.num_params = self.num_vars
        self.sample_tensor = False
        self.is_multivariate = False

        if p is None:
            p_unconst = tf.Variable(tf.random_normal(self.shape))
            p = tf.sigmoid(p_unconst)

        self.p = p

    def __str__(self):
        p = self.p.eval()
        return "probability: \n" + p.__str__()

    def sample(self, size=1):
        p = self.p.eval()
        return bernoulli.rvs(p, size=size)

    def log_prob_idx(self, idx, xs):
        full_idx = (slice(0, None), ) + idx # slice over batch size
        return bernoulli.logpmf(xs[full_idx], self.p[idx])

    def entropy(self):
        return tf.reduce_sum(bernoulli.entropy(self.p))

class Beta(Distribution):
    """Beta

    See :class:`edward.stats.distributions.Beta`
    """
    def __init__(self, shape=1, alpha=None, beta=None):
        Distribution.__init__(self, shape)
        self.num_params = 2*self.num_vars
        self.sample_tensor = False
        self.is_multivariate = False

        if alpha is None:
            alpha_unconst = tf.Variable(tf.random_normal(self.shape))
            alpha = tf.nn.softplus(alpha_unconst)

        if beta is None:
            beta_unconst = tf.Variable(tf.random_normal(self.shape))
            beta = tf.nn.softplus(beta_unconst)

        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return "shape: \n" + a.__str__() + "\n" + \
               "scale: \n" + b.__str__()

    def sample(self, size=1):
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return beta.rvs(a, b, size=size)

    def log_prob_idx(self, idx, xs):
        full_idx = (slice(0, None), ) + idx # slice over batch size
        return beta.logpdf(xs[full_idx], self.alpha[idx], self.beta[idx])

    def entropy(self):
        return tf.reduce_sum(beta.entropy(self.alpha, self.beta))

class Dirichlet(Distribution):
    """Dirichlet

    See :class:`edward.stats.distributions.Dirichlet`
    """
    def __init__(self, shape, alpha=None):
        Distribution.__init__(self, shape)
        self.num_params = self.num_vars
        self.sample_tensor = False
        self.is_multivariate = True

        if alpha is None:
            alpha_unconst = tf.Variable(tf.random_normal(self.shape))
            alpha = tf.nn.softplus(alpha_unconst)

        self.alpha = alpha

    def __str__(self):
        alpha = self.alpha.eval()
        return "concentration: \n" + alpha.__str__()

    def sample(self, size=1):
        alpha = self.alpha.eval()
        return dirichlet.rvs(alpha, size=size)

    def log_prob_idx(self, idx, xs):
        """
        ``log p(xs[:, idx, :] | params[idx, :])``
        where ``idx`` is of dimension ``shape[:-1]``
        """
        idx = idx + (slice(0, None), ) # slice over multivariate dimension
        full_idx = (slice(0, None), ) + idx # slice over batch size
        return dirichlet.logpdf(xs[full_idx], self.alpha[idx])

    def entropy(self):
        return tf.reduce_sum(dirichlet.entropy(self.alpha))

class InvGamma(Distribution):
    """Inverse Gamma

    See :class:`edward.stats.distributions.InvGamma`
    """
    def __init__(self, shape=1, alpha=None, beta=None):
        Distribution.__init__(self, shape)
        self.num_params = 2*self.num_vars
        self.sample_tensor = False
        self.is_multivariate = False

        if alpha is None:
            alpha_unconst = tf.Variable(tf.random_normal(self.shape))
            alpha = tf.nn.softplus(alpha_unconst) + 1e-2

        if beta is None:
            beta_unconst = tf.Variable(tf.random_normal(self.shape))
            beta = tf.nn.softplus(beta_unconst) + 1e-2

        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return "shape: \n" + a.__str__() + "\n" + \
               "scale: \n" + b.__str__()

    def sample(self, size=1):
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return invgamma.rvs(a, b, size=size)

    def log_prob_idx(self, idx, xs):
        full_idx = (slice(0, None), ) + idx # slice over batch size
        return invgamma.logpdf(xs[full_idx], self.alpha[idx], self.beta[idx])

    def entropy(self):
        return tf.reduce_sum(invgamma.entropy(self.alpha, self.beta))

class Multinomial(Distribution):
    """Multinomial

    See :class:`edward.stats.distributions.Multinomial`

    ``p(x | params ) = prod_{idx in shape[:-1]} Multinomial(x[idx] | pi[idx])``

    where ``x[idx]`` represents a multivariate random variable, and 
    ``params = pi.shape[-1]`` denotes the multivariate dimension.

    Notes
    -----
    For each slice along the last dimension (each multinomial
    distribution), it assumes a single trial (n=1) when sampling and
    calculating the density.
    """
    def __init__(self, shape, pi=None):
        if shape[-1] == 1:
            raise ValueError("Multinomial is not supported for K=1. Use Bernoulli.")

        Distribution.__init__(self, shape)
        self.num_params = np.prod(shape[:-1]) * (shape[-1] -1)
        self.sample_tensor = False
        self.is_multivariate = True

        if pi is None:
            real_shape = self.shape[:-1]
            K_minus_one = self.shape[-1] - 1
            pi_unconst = tf.Variable(tf.random_normal([real_shape + (K_minus_one, )]))
            pi = to_simplex(pi_unconst)

        self.pi = pi

    def __str__(self):
        pi = self.pi.eval()
        return "probability: \n" + pi.__str__()

    def sample(self, size=1):
        pi = self.pi.eval()
        return multinomial.rvs(np.ones(self.shape[:-1]), pi, size=size)

    def log_prob_idx(self, idx, xs):
        """
        ``log p(xs[:, idx, :] | params[idx, :])``
        where ``idx`` is of dimension ``shape[:-1]``
        """
        idx_K = idx + (slice(0, None), ) # slice over multivariate dimension
        full_idx = (slice(0, None), ) + idx_K # slice over batch size
        return multinomial.logpmf(xs[full_idx], np.ones(self.shape[:-1])[idx], self.pi[idx_K])

    def entropy(self):
        return tf.reduce_sum(multinomial.entropy(np.ones(self.shape[:-1]), self.pi))

class Normal(Distribution):
    """Normal 

    See :class:`edward.stats.distributions.Norm`
    """
    def __init__(self, shape=1, loc=None, scale=None):
        Distribution.__init__(self, shape)
        self.num_params = 2*self.num_vars
        self.sample_tensor = True
        self.is_multivariate = False

        if loc is None:
            loc = tf.Variable(tf.random_normal(self.shape))

        if scale is None:
            scale_unconst = tf.Variable(tf.random_normal(self.shape))
            scale = tf.nn.softplus(scale_unconst)

        self.loc = loc
        self.scale = scale

    def __str__(self):
        sess = get_session()
        m, s = sess.run([self.loc, self.scale])
        return "mean: \n" + m.__str__() + "\n" + \
               "std dev: \n" + s.__str__()

    def sample_noise(self, size=1):
        return tf.random_normal((size, ) + self.shape)

    def reparam(self, eps):
        return self.loc + eps * self.scale

    def log_prob_idx(self, idx, xs):
        full_idx = (slice(0, None), ) + idx # slice over batch size
        return norm.logpdf(xs[full_idx], self.loc[idx], self.scale[idx])

    def entropy(self):
        return tf.reduce_sum(norm.entropy(scale=self.scale))

class PointMass(Distribution):
    """Point mass distribution

    ``p(x | params ) = prod_{idx in shape} Dirac(x[idx] | params[idx])``

    ``Dirac(x; p)`` is the Dirac delta distribution with density equal to
    1 if x == p and 0 otherwise.

    Parameters
    ----------
    params : np.ndarray, tf.Variable, optional
        if not specified, everything initialized to :math:`\mathcal{N}(0,1)`
    """
    def __init__(self, shape=1, params=None):
        Distribution.__init__(self, shape)
        self.num_params = self.num_vars
        self.sample_tensor = True
        self.is_multivariate = False

        if params is None:
            params = tf.Variable(tf.random_normal(self.shape))

        self.params = params

    def __str__(self):
        if self.params.get_shape()[0] == 0:
            return "parameter values: \n" + "None"

        params = self.params.eval()
        return "parameter values: \n" + params.__str__()

    def sample(self, size=1):
        """Sample from a point mass distribution.
        
        Each sample is simply the set of point masses, as all
        probability mass is located there.

        Parameters
        ----------
        size: int
            number of samples        
        """
        return tf.pack([self.params]*size)

    def log_prob_idx(self, idx, xs):
        """Log probability at an index of a point mass distribution.

        Returns
        -------
        tf.Tensor
            A vector where the jth element is 1 if xs[j, idx] is equal
            to params[idx], 0 otherwise.
        """
        full_idx = (slice(0, None), ) + idx # slice over batch size
        return tf.cast(tf.equal(xs[full_idx], self.params[idx]), dtype=tf.float32)
