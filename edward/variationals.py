from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import bernoulli, beta, norm, dirichlet, invgamma, multinomial
from edward.util import Variable

class Variational:
    """A stack of variational families."""
    def __init__(self, layers=[]):
        self.layers = layers
        self.num_factors = 0
        self.num_vars = 0
        self.num_params = 0
        self.is_reparam = True

    def add(self, layer):
        """
        Adds a layer instance on top of the layer stack.

        Parameters
        ----------
            layer: layer instance.
        """
        self.layers += [layer]
        self.num_factors += layer.num_factors
        self.num_vars += layer.num_vars
        self.num_params += layer.num_params
        self.is_reparam = self.is_reparam and 'reparam' in layer.__class__.__dict__

    def mapping(self, x):
        return [layer.mapping(x) for layer in self.layers]

    def set_params(self, params):
        [layer.set_params(params[i]) for i,layer in enumerate(self.layers)]

    def print_params(self, sess):
        [layer.print_params(sess) for layer in self.layers]

    def sample_noise(self, size):
        eps_layers = [layer.sample_noise(size) for layer in self.layers]
        return np.concatenate(eps_layers, axis=1)

    def reparam(self, eps):
        z_layers = []
        start = final = 0
        for layer in self.layers:
            final += layer.num_vars
            z_layers += [layer.reparam(eps[:, start:final])]
            start = final

        return tf.concat(1, z_layers)

    def sample(self, size, sess):
        #z_layers = [layer.sample(size, sess) for layer in self.layers]
        # This is temporary to deal with reparameterizable ones.
        z_layers = []
        for layer in self.layers:
            z_layer = layer.sample(size, sess)
            if isinstance(layer, Normal):
                z_layer = sess.run(z_layer)

            z_layers += [z_layer]

        return np.concatenate(z_layers, axis=1)

    def log_prob_zi(self, i, zs):
        start = final = 0
        for layer in self.layers:
            final += layer.num_vars
            if i < layer.num_factors:
                return layer.log_prob_zi(i, zs[:, start:final])

            i = i - layer.num_factors
            start = final

        raise IndexError()

class Likelihood:
    """
    Base class for variational likelihoods, q(z | lambda).

    Parameters
    ----------
    num_factors : int
        Number of factors.
    """
    def __init__(self, num_factors):
        self.num_factors = num_factors
        self.num_vars = None # number of posterior latent variables
        self.num_params = None # number of local variational parameters
        # TODO attribute for number of global variational parameters

    def mapping(self, x):
        """
        A mapping from data point x -> lambda, the local variational
        parameters, which are parameters specific to x.

        Parameters
        ----------
        x : Data
            Data point

        Returns
        -------
        list
            A list of TensorFlow tensors, where each element is a
            particular set of local parameters.

        Notes
        -----
        In classical variational inference, the mapping can be
        interpreted as the collection of all local variational
        parameters; the output is simply the projection to the
        relevant subset of local parameters.

        For local variational parameters with constrained support, the
        mapping additionally acts as a transformation. The parameters
        to be optimized live on the unconstrained space; the output of
        the mapping is then constrained variational parameters.

        Global parameterizations are useful to prevent the parameters
        of this mapping to grow with the number of data points, and
        also as an implicit regularization. This is known as inverse
        mappings in Helmholtz machines and variational auto-encoders,
        and parameter tying in message passing. The mapping is a
        function of data point with a fixed number of parameters, and
        it tries to (in some sense) "predict" the best local
        variational parameters given this lower rank.
        """
        raise NotImplementedError()

    def set_params(self, params):
        """
        This sets the parameters of the variational family, for use in
        other methods of the class.

        Parameters
        ----------
        params : list
            Each element in the list is a particular set of local parameters.
        """
        raise NotImplementedError()

    # TODO use __str__(self):
    def print_params(self, sess):
        raise NotImplementedError()

    def sample_noise(self, size):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)

        Returns
        -------
        np.ndarray
            size x dim(lambda) array of type np.float32, where each
            row is a sample from q.

        Notes
        -----
        Unlike the other methods, this return object is a realization
        of a TensorFlow array. This is required as we rely on
        NumPy/SciPy for sampling from distributions.
        """
        raise NotImplementedError()

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        raise NotImplementedError()

    def sample(self, size, sess=None):
        """
        z ~ q(z | lambda)

        Parameters
        ----------
        sess : tf.Session, optional

        Returns
        -------
        np.ndarray
            size x dim(z) array of type np.float32, where each
            row is a sample from q.

        Notes
        -----
        Unlike the other methods, this return object is a realization
        of a TensorFlow array. This is required as we rely on
        NumPy/SciPy for sampling from distributions.

        The method defaults to sampling noise and reparameterizing it
        (which will raise an error if this is not possible).
        """
        return self.reparam(self.sample_noise(size))

    def log_prob_zi(self, i, zs):
        """
        log q(z_i | lambda)
        Note this calculates the density of the ith factor, not
        necessarily the ith latent variable (such as for multivariate
        factors).

        Parameters
        ----------
        i : int
            Index of the factor to take the log density of.
        zs : np.array
            n_minibatch x num_vars

        Returns
        -------
        [log q(zs[1]_i | lambda), ..., log q(zs[S]_i | lambda)]
        """
        raise NotImplementedError()

class Bernoulli(Likelihood):
    """
    q(z | lambda) = prod_{i=1}^d Bernoulli(z[i] | p[i])
    where lambda = p.
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_vars = self.num_factors
        if self.num_vars == 1:
            self.num_params = self.num_factors
        else:
            self.num_params = self.num_factors - 1

        self.p = None

    def mapping(self, x):
        # Transform unconstrained parameters to lie on simplex.
        p = Variable("p", [self.num_params])
        p_const = tf.sigmoid(p)
        if self.num_vars > 1:
            p_const = tf.concat(0,
                [p_const, tf.expand_dims(1.0 - tf.reduce_sum(p_const), 0)])

        return [p_const]

    def set_params(self, params):
        self.p = params[0]

    def print_params(self, sess):
        p = sess.run(self.p)
        print("probability:")
        print(p)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        p = sess.run(self.p)
        z = np.zeros((size, self.num_vars))
        for d in range(self.num_vars):
            z[:, d] = bernoulli.rvs(p[d], size=size)

        return z

    def log_prob_zi(self, i, zs):
        """log q(z_i | lambda)"""
        if i >= self.num_factors:
            raise IndexError()

        return bernoulli.logpmf(zs[:, i], self.p[i])

class Beta(Likelihood):
    """
    q(z | lambda) = prod_{i=1}^d Beta(z[i] | a[i], b[i])
    where lambda = {a, b}.
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.a = None
        self.b = None

    def mapping(self, x):
        alpha = Variable("alpha", [self.num_vars])
        beta = Variable("beta", [self.num_vars])
        return [tf.nn.softplus(alpha), tf.nn.softplus(beta)]

    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]

    def print_params(self, sess):
        a, b = sess.run([self.a, self.b])
        print("shape:")
        print(a)
        print("scale:")
        print(b)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([self.a, self.b])
        z = np.zeros((size, self.num_vars))
        for d in range(self.num_vars):
            z[:, d] = beta.rvs(a[d], b[d], size=size)

        return z

    def log_prob_zi(self, i, zs):
        """log q(z_i | lambda)"""
        if i >= self.num_factors:
            raise IndexError()

        return beta.logpdf(zs[:, i], self.a[i], self.b[i])

class Dirichlet(Likelihood):
    """
    q(z | lambda) = prod_{i=1}^d Dirichlet(z_i | alpha[i, :])
    where z is a flattened vector such that z_i represents
    the ith factor z[(i-1)*K:i*K], and lambda = alpha.
    """
    def __init__(self, num_factors, K):
        Likelihood.__init__(self, num_factors)
        self.num_vars = K*num_factors
        self.num_params = K*num_factors
        self.K = K # dimension of each factor
        self.alpha = None

    def mapping(self, x):
        alpha = Variable("dirichlet_alpha", [self.num_factors, self.K])
        return [tf.nn.softplus(alpha)]

    def set_params(self, params):
        self.alpha = params[0]

    def print_params(self, sess):
        alpha = sess.run(self.alpha)
        print("concentration vector:")
        print(alpha)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        alpha = sess.run(self.alpha)
        z = np.zeros((size, self.num_vars))
        for i in xrange(self.num_factors):
            z[:, (i*self.K):((i+1)*self.K)] = dirichlet.rvs(alpha[i, :],
                                                            size=size)

        return z

    def log_prob_zi(self, i, zs):
        """log q(z_i | lambda)"""
        # Note this calculates the log density with respect to z_i,
        # which is the ith factor and not the ith latent variable.
        if i >= self.num_factors:
            raise IndexError()

        return dirichlet.logpdf(zs[:, (i*self.K):((i+1)*self.K)],
                                self.alpha[i, :])

class InvGamma(Likelihood):
    """
    q(z | lambda) = prod_{i=1}^d Inv_Gamma(z[i] | a[i], b[i])
    where lambda = {a, b}.
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.a = None
        self.b = None

    def mapping(self, x):
        alpha = Variable("alpha", [self.num_vars])
        beta = Variable("beta", [self.num_vars])
        return [tf.nn.softplus(alpha), tf.nn.softplus(beta)]

    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]

    def print_params(self, sess):
        a, b = sess.run([self.a, self.b])
        print("shape:")
        print(a)
        print("scale:")
        print(b)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        a, b = sess.run([self.a, self.b])
        z = np.zeros((size, self.num_vars))
        for d in range(self.num_vars):
            z[:, d] = invgamma.rvs(a[d], b[d], size=size)

        return z

    def log_prob_zi(self, i, zs):
        """log q(z_i | lambda)"""
        if i >= self.num_factors:
            raise IndexError()

        return invgamma.logpdf(zs[:, i], self.a[i], self.b[i])

class Multinomial(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Multinomial(z_i | pi[i, :])
    where z is a flattened vector such that z_i represents
    the ith factor z[(i-1)*K:i*K], and lambda = alpha.

    Notes
    -----
    For each factor (multinomial distribution), it assumes a single
    trial (n=1) when sampling and calculating the density.
    """
    def __init__(self, num_factors, K):
        if K == 1:
            raise ValueError("Multinomial is not supported for K=1. Use Bernoulli.")

        Likelihood.__init__(self, num_factors)
        self.num_vars = K*num_factors
        self.num_params = K*num_factors
        self.K = K # dimension of each factor
        self.pi = None

    def mapping(self, x):
        # Transform unconstrained parameters to lie on simplex.
        pi = Variable("pi", [self.num_factors, self.K-1])
        pi_const = tf.sigmoid(pi)
        pi_const = tf.concat(1,
            [pi_const, tf.expand_dims(1.0 - tf.reduce_sum(pi_const), 0)])

        return [pi_const]

    def set_params(self, params):
        self.pi = params[0]

    def print_params(self, sess):
        pi = sess.run(self.pi)
        print("probability vector:")
        print(pi)

    def sample(self, size, sess):
        """z ~ q(z | lambda)"""
        pi = sess.run(self.pi)
        z = np.zeros((size, self.num_vars))
        for i in xrange(self.num_factors):
            z[:, (i*self.K):((i+1)*self.K)] = multinomial.rvs(1, pi[i, :],
                                                              size=size)

        return z

    def log_prob_zi(self, i, zs):
        """log q(z_i | lambda)"""
        # Note this calculates the log density with respect to z_i,
        # which is the ith factor and not the ith latent variable.
        if i >= self.num_factors:
            raise IndexError()

        return multinomial.logpmf(zs[:, (i*self.K):((i+1)*self.K)],
                                  1, self.pi[i, :])

class Normal(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Normal(z[i] | m[i], s[i])
    where lambda = {m, s}.
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.m = None
        self.s = None

    def mapping(self, x):
        mean = Variable("mu", [self.num_vars])
        stddev = Variable("sigma", [self.num_vars])
        return [tf.identity(mean), tf.nn.softplus(stddev)]

    def set_params(self, params):
        self.m = params[0]
        self.s = params[1]

    def print_params(self, sess):
        m, s = sess.run([self.m, self.s])
        print("mean:")
        print(m)
        print("std dev:")
        print(s)

    def sample_noise(self, size):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        # Not using this, since TensorFlow has a large overhead
        # whenever calling sess.run().
        #samples = sess.run(tf.random_normal(self.samples.get_shape()))
        return norm.rvs(size=(size, self.num_vars))

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        return self.m + eps * self.s

    def log_prob_zi(self, i, zs):
        """log q(z_i | lambda)"""
        if i >= self.num_factors:
            raise IndexError()

        mi = self.m[i]
        si = self.s[i]
        return norm.logpdf(zs[:, i], mi, si)

    # TODO entropy is bugged
    #def entropy(self):
    #    return norm.entropy(self.transform_s(self.s_unconst))

class PointMass(Likelihood):
    """
    Point mass variational family
    """
    def __init__(self, num_vars, transform=tf.identity):
        Likelihood.__init__(self, 1)
        self.num_vars = num_vars
        self.num_params = num_vars
        self.transform = transform
        self.params = None

    def mapping(self, x):
        params = Variable("params", [self.num_vars])
        return [self.transform(params)]

    def set_params(self, params):
        self.params = params[0]

    def print_params(self, sess):
        params = sess.run(self.params)
        print("parameter values:")
        print(params)

    def get_params(self):
        # Return a matrix to be compatible with probability model
        # methods which assume the input is possibly a mini-batch of
        # parameter samples (used for black box variational methods).
        return tf.reshape(self.params, [1, self.num_params])
