from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import bernoulli, beta, norm, dirichlet, invgamma, multinomial
from edward.util import cumprod, get_session, Variable

class Variational:
    """A stack of distribution objects."""
    def __init__(self, layers=[]):
        get_session()
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

    def __str__(self):
        string = ""
        for i in range(len(self.layers)):
            if i != 0:
                string += "\n"

            layer = self.layers[i]
            string += layer.__str__()

        return string

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

    def sample(self, size=1):
        """
        Draws a mix of tensors and placeholders, corresponding to
        TensorFlow-based samplers and SciPy-based samplers depending
        on the distribution factor.

        Parameters
        ----------
        size : int, optional

        Returns
        -------
        tf.Tensor, list
            A tensor concatenating sample outputs of tensors and
            placeholders. The list used to form the tensor is also
            returned so that other procedures can feed values into the
            placeholders.
        """
        samples = []
        for layer in self.layers:
            if layer.sample_tensor:
                samples += [layer.sample(size)]
            else:
                samples += [tf.placeholder(tf.float32, (size, layer.num_vars))]

        return tf.concat(1, samples), samples

    def np_sample(self, samples, size=1):
        """
        Form dictionary to feed any placeholders with np.array
        samples.
        """
        feed_dict = {}
        for sample,layer in zip(samples, self.layers):
            if sample.name.startswith('Placeholder'):
                feed_dict[sample] = layer.sample(size)

        return feed_dict

    def log_prob_i(self, i, xs):
        start = final = 0
        for layer in self.layers:
            final += layer.num_vars
            if i < layer.num_factors:
                return layer.log_prob_i(i, xs[:, start:final])

            i = i - layer.num_factors
            start = final

        raise IndexError()

    def entropy(self):
        out = tf.constant(0.0, dtype=tf.float32)
        for layer in self.layers:
            out += layer.entropy()

        return out

class Distribution:
    """
    Base class for distributions, p(x | params).

    Parameters
    ----------
    num_factors : int
        Number of factors. Default is 1.
    """
    def __init__(self, num_factors=1):
        get_session()
        self.num_factors = num_factors
        self.num_vars = None # number of random variables
        self.num_params = None # number of parameters
        self.sample_tensor = False

    def sample_noise(self, size=1):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)

        Returns
        -------
        np.ndarray
            size x dim(params) array of type np.float32, where each
            row is a sample from p.

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
        s.t. x = reparam(eps; params) ~ p(x | params)
        """
        raise NotImplementedError()

    def sample(self, size=1):
        """
        x ~ p(x | params)

        Returns
        -------
        np.ndarray
            size x dim(x) array of type np.float32, where each
            row is a sample from p.

        Notes
        -----
        Unlike the other methods, this return object is a realization
        of a TensorFlow array. This is required as we rely on
        NumPy/SciPy for sampling from distributions.

        The method defaults to sampling noise and reparameterizing it
        (which will raise an error if this is not possible).
        """
        return self.reparam(self.sample_noise(size))

    def log_prob_i(self, i, xs):
        """
        log p(x_i | params)
        Note this calculates the density of the ith factor, not
        necessarily the ith latent variable (such as for multivariate
        factors).

        Parameters
        ----------
        i : int
            Index of the factor to take the log density of.
        xs : np.array
            n_minibatch x num_vars

        Returns
        -------
        [log p(xs[1]_i | params), ..., log p(xs[S]_i | params)]
        """
        raise NotImplementedError()

    def entropy(self):
        """
        H(p(x| params))
        = E_{p(x | params)} [ - log p(x | params) ]
        = sum_{i=1}^d E_{p(x_i | params)} [ - log p(x_i | params) ]

        Returns
        -------
        tf.Tensor
            scalar
        """
        raise NotImplementedError()

class Bernoulli(Distribution):
    """
    p(x | params) = prod_{i=1}^d Bernoulli(x[i] | p[i])
    where params = p.
    """
    def __init__(self, num_factors=1, p=None):
        Distribution.__init__(self, num_factors)
        self.num_vars = self.num_factors
        self.num_params = self.num_factors
        self.sample_tensor = False

        if p is None:
            p_unconst = Variable("p", [self.num_params])
            p = tf.sigmoid(p_unconst)

        self.p = p

    def __str__(self):
        p = self.p.eval()
        return "probability: \n" + p.__str__()

    def sample(self, size=1):
        """x ~ p(x | params)"""
        p = self.p.eval()
        x = np.zeros((size, self.num_vars))
        for d in range(self.num_vars):
            x[:, d] = bernoulli.rvs(p[d], size=size)

        return x

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        if i >= self.num_factors:
            raise IndexError()

        return bernoulli.logpmf(xs[:, i], self.p[i])

    def entropy(self):
        return tf.reduce_sum(bernoulli.entropy(self.p))

class Beta(Distribution):
    """
    p(x | params) = prod_{i=1}^d Beta(x[i] | alpha[i], beta[i])
    where params = {alpha, beta}.
    """
    def __init__(self, num_factors=1, alpha=None, beta=None):
        Distribution.__init__(self, num_factors)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.sample_tensor = False

        if alpha is None:
            alpha_unconst = Variable("alpha", [self.num_vars])
            alpha = tf.nn.softplus(alpha_unconst)

        if beta is None:
            beta_unconst = Variable("beta", [self.num_vars])
            beta = tf.nn.softplus(beta_unconst)

        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return "shape: \n" + a.__str__() + "\n" + \
               "scale: \n" + b.__str__()

    def sample(self, size=1):
        """x ~ p(x | params)"""
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        x = np.zeros((size, self.num_vars))
        for d in range(self.num_vars):
            x[:, d] = beta.rvs(a[d], b[d], size=size)

        return x

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        if i >= self.num_factors:
            raise IndexError()

        return beta.logpdf(xs[:, i], self.alpha[i], self.beta[i])

    def entropy(self):
        return tf.reduce_sum(beta.entropy(self.alpha, self.beta))

class Dirichlet(Distribution):
    """
    p(x | params) = prod_{i=1}^d Dirichlet(x_i | alpha[i, :])
    where x is a flattened vector such that x_i represents
    the ith factor x[(i-1)*K:i*K], and params = alpha.
    """
    def __init__(self, shape, alpha=None):
        num_factors = shape[0]
        K = shape[-1]
        Distribution.__init__(self, num_factors)
        self.num_vars = K*num_factors
        self.num_params = K*num_factors
        self.K = K # dimension of each factor
        self.sample_tensor = False

        if alpha is None:
            alpha_unconst = Variable("dirichlet_alpha", [self.num_factors, self.K])
            alpha = tf.nn.softplus(alpha_unconst)

        self.alpha = alpha

    def __str__(self):
        alpha = self.alpha.eval()
        return "concentration vector: \n" + alpha.__str__()

    def sample(self, size=1):
        """x ~ p(x | params)"""
        alpha = self.alpha.eval()
        x = np.zeros((size, self.num_vars))
        for i in range(self.num_factors):
            x[:, (i*self.K):((i+1)*self.K)] = dirichlet.rvs(alpha[i, :],
                                                            size=size)

        return x

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        # Note this calculates the log density with respect to x_i,
        # which is the ith factor and not the ith latent variable.
        if i >= self.num_factors:
            raise IndexError()

        return dirichlet.logpdf(xs[:, (i*self.K):((i+1)*self.K)],
                                self.alpha[i, :])

    def entropy(self):
        return tf.reduce_sum(dirichlet.entropy(self.alpha))

class InvGamma(Distribution):
    """
    p(x | params) = prod_{i=1}^d Inv_Gamma(x[i] | alpha[i], beta[i])
    where params = {alpha, beta}.
    """
    def __init__(self, num_factors=1, alpha=None, beta=None):
        Distribution.__init__(self, num_factors)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.sample_tensor = False

        if alpha is None:
            alpha_unconst = Variable("alpha", [self.num_vars])
            alpha = tf.nn.softplus(alpha_unconst) + 1e-2

        if beta is None:
            beta_unconst = Variable("beta", [self.num_vars])
            beta = tf.nn.softplus(beta_unconst) + 1e-2

        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return "shape: \n" + a.__str__() + "\n" + \
               "scale: \n" + b.__str__()

    def sample(self, size=1):
        """x ~ p(x | params)"""
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        x = np.zeros((size, self.num_vars))
        for d in range(self.num_vars):
            x[:, d] = invgamma.rvs(a[d], b[d], size=size)

        return x

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        if i >= self.num_factors:
            raise IndexError()

        return invgamma.logpdf(xs[:, i], self.alpha[i], self.beta[i])

    def entropy(self):
        return tf.reduce_sum(invgamma.entropy(self.alpha, self.beta))

class Multinomial(Distribution):
    """
    p(x | params ) = prod_{i=1}^d Multinomial(x_i | pi[i, :])
    where x is a flattened vector such that x_i represents
    the ith factor x[(i-1)*K:i*K], and params = alpha.

    Notes
    -----
    For each factor (multinomial distribution), it assumes a single
    trial (n=1) when sampling and calculating the density.
    """
    def __init__(self, shape, pi=None):
        num_factors = shape[0]
        K = shape[-1]
        if K == 1:
            raise ValueError("Multinomial is not supported for K=1. Use Bernoulli.")

        Distribution.__init__(self, num_factors)
        self.num_vars = K*num_factors
        self.num_params = K*num_factors
        self.K = K # dimension of each factor
        self.sample_tensor = False

        if pi is None:
            # Transform a real (K-1)-vector to K-dimensional simplex.
            pi_unconst = Variable("pi", [self.num_factors, self.K-1])
            eq = -tf.log(tf.cast(self.K - 1 - tf.range(self.K-1), dtype=tf.float32))
            x = tf.sigmoid(eq + pi_unconst)
            pil = tf.concat(1, [x, tf.ones([self.num_factors, 1])])
            piu = tf.concat(1, [tf.ones([self.num_factors, 1]), 1.0 - x])
            # cumulative product along 1st axis
            S = tf.pack([cumprod(piu_x) for piu_x in tf.unpack(piu)])
            pi = S * pil

        self.pi = pi

    def __str__(self):
        pi = self.pi.eval()
        return "probability vector: \n" + pi.__str__()

    def sample(self, size=1):
        """x ~ p(x | params)"""
        pi = self.pi.eval()
        x = np.zeros((size, self.num_vars))
        for i in range(self.num_factors):
            x[:, (i*self.K):((i+1)*self.K)] = multinomial.rvs(1, pi[i, :],
                                                              size=size)

        return x

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        # Note this calculates the log density with respect to x_i,
        # which is the ith factor and not the ith latent variable.
        if i >= self.num_factors:
            raise IndexError()

        return multinomial.logpmf(xs[:, (i*self.K):((i+1)*self.K)],
                                  1, self.pi[i, :])

    def entropy(self):
        return tf.reduce_sum(multinomial.entropy(1, self.pi))

class Normal(Distribution):
    """
    p(x | params ) = prod_{i=1}^d Normal(x[i] | loc[i], scale[i])
    where params = {loc, scale}.
    """
    def __init__(self, num_factors=1, loc=None, scale=None):
        Distribution.__init__(self, num_factors)
        self.num_vars = self.num_factors
        self.num_params = 2*self.num_factors
        self.sample_tensor = True

        if loc is None:
            loc = Variable("mu", [self.num_vars])

        if scale is None:
            scale_unconst = Variable("sigma", [self.num_vars])
            scale = tf.nn.softplus(scale_unconst)

        self.loc = loc
        self.scale = scale

    def __str__(self):
        sess = get_session()
        m, s = sess.run([self.loc, self.scale])
        return "mean: \n" + m.__str__() + "\n" + \
               "std dev: \n" + s.__str__()

    def sample_noise(self, size=1):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)
        """
        return tf.random_normal((size, self.num_vars))

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)
        """
        return self.loc + eps * self.scale

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        if i >= self.num_factors:
            raise IndexError()

        loci = self.loc[i]
        scalei = self.scale[i]
        return norm.logpdf(xs[:, i], loci, scalei)

    def entropy(self):
        return tf.reduce_sum(norm.entropy(scale=self.scale))

class PointMass(Distribution):
    """
    Point mass distribution

    p(x | params ) = prod_{i=1}^d Dirac(x[i] | params[i])

    Dirac(x; p) is the Dirac delta distribution with density equal to
    1 if x == p and 0 otherwise.
    """
    def __init__(self, num_vars=1, params=None):
        Distribution.__init__(self, 1)
        self.num_vars = num_vars
        self.num_params = num_vars
        self.sample_tensor = True

        if params is None:
            params = Variable("params", [self.num_vars])

        self.params = params

    def __str__(self):
        if self.params.get_shape()[0] == 0:
            return "parameter values: \n" + "None"

        params = self.params.eval()
        return "parameter values: \n" + params.__str__()

    def sample(self, size=1):
        # Return a matrix where each row is the same set of
        # parameters. This is to be compatible with probability model
        # methods which assume the input is possibly a mini-batch of
        # parameter samples (as in black box variational methods).
        return tf.pack([self.params]*size)

    def log_prob_i(self, i, xs):
        """log p(x_i | params)"""
        if i >= self.num_factors:
            raise IndexError()

        # a vector where the jth element is 1 if xs[j, i] is equal to
        # the ith parameter, 0 otherwise
        return tf.cast(tf.equal(xs[:, i], self.params[i]), dtype=tf.float32)
