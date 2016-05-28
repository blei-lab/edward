from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import bernoulli, beta, norm, dirichlet, invgamma, multinomial
from edward.util import cumprod, Variable

class Variational:
    """A stack of variational families."""
    def __init__(self, layers=[]):
        self.layers = layers
        if layers == []:
            self.num_factors = 0
            self.num_vars = 0
            self.num_params = 0
            self.is_reparam = True
            self.is_normal = True
            self.is_entropy = True
            self.sample_tensor = []
        else:
            self.num_factors = sum([layer.num_factors for layer in self.layers])
            self.num_vars = sum([layer.num_vars for layer in self.layers])
            self.num_params = sum([layer.num_params for layer in self.layers])
            self.is_reparam = all(['reparam' in layer.__class__.__dict__
                                   for layer in self.layers])
            self.is_normal = all([isinstance(layer, Normal)
                                  for layer in self.layers])
            self.is_entropy = all(['entropy' in layer.__class__.__dict__
                                   for layer in self.layers])
            self.sample_tensor = [layer.sample_tensor for layer in self.layers]

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
        self.is_entropy = self.is_entropy and 'entropy' in layer.__class__.__dict__
        self.is_normal = self.is_normal and isinstance(layer, Normal)
        self.sample_tensor += [layer.sample_tensor]

    def sample(self, x, size=1):
        """
        Draws a mix of tensors and placeholders, corresponding to
        TensorFlow-based samplers and SciPy-based samplers depending
        on the variational factor.

        Parameters
        ----------
        x : Data
        size : int, optional

        Returns
        -------
        tf.Tensor, list
            A tensor concatenating sample outputs of tensors and
            placeholders. The list used to form the tensor is also
            returned so that other procedures can feed values into the
            placeholders.

        Notes
        -----
        This sets parameters of the variational distribution according
        to any data points it conditions on. This means after calling
        this method, log_prob_zi() is well-defined, even though
        mathematically it is a function of both z and x, and it only
        takes z as input.
        """
        self._set_params(self._mapping(x))
        samples = []
        for layer in self.layers:
            if layer.sample_tensor:
                samples += [layer.sample(size)]
            else:
                samples += [tf.placeholder(tf.float32, (size, layer.num_vars))]

        return tf.concat(1, samples), samples

    def np_sample(self, samples, size=1, sess=None):
        """
        Form dictionary to feed any placeholders with np.array
        samples.
        """
        feed_dict = {}
        for sample,layer in zip(samples, self.layers):
            if sample.name.startswith('Placeholder'):
                feed_dict[sample] = layer.sample(size, sess)

        return feed_dict

    def print_params(self, sess):
        [layer.print_params(sess) for layer in self.layers]

    def log_prob_zi(self, i, zs):
        start = final = 0
        for layer in self.layers:
            final += layer.num_vars
            if i < layer.num_factors:
                return layer.log_prob_zi(i, zs[:, start:final])

            i = i - layer.num_factors
            start = final

        raise IndexError()

    def entropy(self):
        out = tf.constant(0.0, dtype=tf.float32)
        for layer in self.layers:
            out += layer.entropy()

        return out

    def _mapping(self, x):
        return [layer.mapping(x) for layer in self.layers]

    def _set_params(self, params):
        [layer.set_params(params[i]) for i,layer in enumerate(self.layers)]

class Likelihood:
    """
    Base class for variational likelihoods, q(z | lambda).

    Parameters
    ----------
    num_factors : int
        Number of factors. Default is 1.
    """
    def __init__(self, num_factors=1):
        self.num_factors = num_factors
        self.num_vars = None # number of posterior latent variables
        self.num_params = None # number of variational parameters
        self.sample_tensor = False

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

    def print_params(self, sess):
        raise NotImplementedError()

    def sample_noise(self, size=1):
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

    def sample(self, size=1, sess=None):
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

    def entropy(self):
        """
        H(q(z| lambda))
        = E_{q(z | lambda)} [ - log q(z | lambda) ]
        = sum_{i=1}^d E_{q(z_i | lambda)} [ - log q(z_i | lambda) ]

        Returns
        -------
        tf.Tensor
            scalar
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
        self.num_params = self.num_factors
        self.sample_tensor = False

        self.p = None

    def mapping(self, x):
        p = Variable("p", [self.num_params])
        return [tf.sigmoid(p)]

    def set_params(self, params):
        self.p = params[0]

    def print_params(self, sess):
        p = sess.run(self.p)
        print("probability:")
        print(p)

    def sample(self, size=1, sess=None):
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

    def entropy(self):
        return tf.reduce_sum(bernoulli.entropy(self.p))

class Beta(Likelihood):
    """
    q(z | lambda) = prod_{i=1}^d Beta(z[i] | a[i], b[i])
    where lambda = {a, b}.
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.sample_tensor = False

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

    def sample(self, size=1, sess=None):
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

    def entropy(self):
        return tf.reduce_sum(beta.entropy(self.a, self.b))

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
        self.sample_tensor = False

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

    def sample(self, size=1, sess=None):
        """z ~ q(z | lambda)"""
        alpha = sess.run(self.alpha)
        z = np.zeros((size, self.num_vars))
        for i in range(self.num_factors):
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

    def entropy(self):
        return tf.reduce_sum(dirichlet.entropy(self.alpha))

class InvGamma(Likelihood):
    """
    q(z | lambda) = prod_{i=1}^d Inv_Gamma(z[i] | a[i], b[i])
    where lambda = {a, b}.
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.sample_tensor = False

        self.a = None
        self.b = None

    def mapping(self, x):
        alpha = Variable("alpha", [self.num_vars])
        beta = Variable("beta", [self.num_vars])
        return [tf.nn.softplus(alpha)+1e-2, tf.nn.softplus(beta)+1e-2]

    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]

    def print_params(self, sess):
        a, b = sess.run([self.a, self.b])
        print("shape:")
        print(a)
        print("scale:")
        print(b)

    def sample(self, size=1, sess=None):
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

    def entropy(self):
        return tf.reduce_sum(invgamma.entropy(self.a, self.b))

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
        self.sample_tensor = False

        self.pi = None

    def mapping(self, x):
        # Transform a real (K-1)-vector to K-dimensional simplex.
        pi = Variable("pi", [self.num_factors, self.K-1])
        eq = -tf.log(tf.cast(self.K - 1 - tf.range(self.K-1), dtype=tf.float32))
        z = tf.sigmoid(eq + pi)
        pil = tf.concat(1, [z, tf.ones([self.num_factors, 1])])
        piu = tf.concat(1, [tf.ones([self.num_factors, 1]), 1.0 - z])
        # cumulative product along 1st axis
        S = tf.pack([cumprod(piu_x) for piu_x in tf.unpack(piu)])
        return [S * pil]

    def set_params(self, params):
        self.pi = params[0]

    def print_params(self, sess):
        pi = sess.run(self.pi)
        print("probability vector:")
        print(pi)

    def sample(self, size=1, sess=None):
        """z ~ q(z | lambda)"""
        pi = sess.run(self.pi)
        z = np.zeros((size, self.num_vars))
        for i in range(self.num_factors):
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

    def entropy(self):
        return tf.reduce_sum(multinomial.entropy(1, self.pi))

class Normal(Likelihood):
    """
    q(z | lambda ) = prod_{i=1}^d Normal(z[i] | m[i], s[i])
    where lambda = {m, s}.
    """
    def __init__(self, *args, **kwargs):
        Likelihood.__init__(self, *args, **kwargs)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.sample_tensor = True

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

    def sample_noise(self, size=1):
        """
        eps = sample_noise() ~ s(eps)
        s.t. z = reparam(eps; lambda) ~ q(z | lambda)
        """
        return tf.random_normal((size, self.num_vars))

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

    def entropy(self):
        return tf.reduce_sum(norm.entropy(scale=self.s))

class PointMass(Likelihood):
    """
    Point mass variational family

    q(z | lambda ) = prod_{i=1}^d Dirac(z[i] | params[i])
    where lambda = params. Dirac(x; p) is the Dirac delta distribution
    with density equal to 1 if x == p and 0 otherwise.
    """
    def __init__(self, num_vars=1, transform=tf.identity):
        Likelihood.__init__(self, 1)
        self.num_vars = num_vars
        self.num_params = num_vars
        self.transform = transform
        self.params = None
        self.sample_tensor = True

    def mapping(self, x):
        params = Variable("params", [self.num_vars])
        return [self.transform(params)]

    def set_params(self, params):
        self.params = params[0]

    def print_params(self, sess):
        params = sess.run(self.params)
        print("parameter values:")
        print(params)

    def sample(self, size=1, sess=None):
        # Return a matrix where each row is the same set of
        # parameters. This is to be compatible with probability model
        # methods which assume the input is possibly a mini-batch of
        # parameter samples (as in black box variational methods).
        return tf.pack([self.params]*size)

    def log_prob_zi(self, i, zs):
        """log q(z_i | lambda)"""
        if i >= self.num_factors:
            raise IndexError()

        # a vector where the jth element is 1 if zs[j, i] is equal to
        # the ith parameter, 0 otherwise
        return tf.cast(tf.equal(zs[:, i], self.params[i]), dtype=tf.float32)
