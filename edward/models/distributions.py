from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import bernoulli, beta, norm, dirichlet, invgamma, multinomial
from edward.util import cumprod, get_session

class Variational:
    """A container for collecting distribution objects."""
    def __init__(self, layers=[]):
        get_session()
        self.layers = layers
        if layers == []:
            self.shape = []
            self.num_vars = 0
            self.num_params = 0
            self.is_reparam = True
            self.is_normal = True
            self.is_entropy = True
            self.sample_tensor = []
        else:
            self.shape = [layer.shape for layer in self.layers]
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
        layer : layer instance.
        """
        self.layers += [layer]
        self.shape += [layer.shape]
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
        on the layer.

        Parameters
        ----------
        size : int, optional

        Returns
        -------
        list or tf.Tensor
            If more than one layer, a list of tensors, one for each
            layer. If one layer, a single tensor. If a layer requires
            SciPy to sample, its corresponding tensor is a
            tf.placeholder.
        """
        samples = []
        for layer in self.layers:
            if layer.sample_tensor:
                samples += [layer.sample(size)]
            elif size == 1:
                samples += [tf.placeholder(tf.float32, (layer.num_vars))]
            else:
                samples += [tf.placeholder(tf.float32, (size, layer.num_vars))]

        if len(samples) == 1:
            samples = samples[0]

        return samples

    def np_dict(self, samples):
        """
        Form dictionary to feed any placeholders with np.array
        samples.

        Parameters
        ----------
        samples : list or tf.Tensor
        """
        if not isinstance(samples, list):
            samples = [samples]

        feed_dict = {}
        for sample,layer in zip(samples, self.layers):
            if sample.name.startswith('Placeholder'):
                size = sample.get_shape()[0]
                feed_dict[sample] = layer.sample(size)

        return feed_dict

    def log_prob(self, xs):
        """
        Parameters
        ----------
        xs : list or tf.Tensor or np.array
            If more than one layer, a list of tf.Tensors or np.array's
            of dimension (batch x shape) or shape. If one layer, a
            tf.Tensor or np.array of (batch x shape) or shape.

        Notes
        -----
        This method may be removed in the future in favor of indexable
        log_prob methods, e.g., for automatic Rao-Blackwellization.
        """
        if len(self.layers) == 1:
            return self.layers[0].log_prob(xs)

        n_minibatch = xs[0].get_shape()[0]
        log_prob = tf.zeros([n_minibatch], dtype=tf.float32)
        for l,layer in enumerate(self.layers):
            # TODO make sure it sums correctly if xs is self.shape
            log_prob += layer.log_prob(xs[l])

        return log_prob

    #def log_prob_idx(self, idx, xs):
    #    """
    #    Parameters
    #    ----------
    #    idx : tuple
    #        (element in list x shape) if more than one layer; shape if
    #        only one layer
    #    xs : list or tf.Tensor or np.array
    #         If more than one layer, a list of tf.Tensors or np.array's
    #         of dimension (batch x shape). If one layer, a tf.Tensor or
    #         np.array of (batch x shape).
    #    """
    #    # TODO implement above
    #    start = final = 0
    #    for layer in self.layers:
    #        final += layer.num_vars
    #        if i < layer.num_factors:
    #            return layer.log_prob_idx(i, xs[:, start:final])

    #        i = i - layer.num_factors
    #        start = final

    #    raise IndexError()

    def entropy(self):
        out = tf.constant(0.0, dtype=tf.float32)
        for layer in self.layers:
            out += layer.entropy()

        return out

class Distribution:
    """
    Base class for distributions

    p(x | params) = prod_{idx in shape} p(x[idx] | params[idx])

    where shape is the shape of x and params[idx] denote the
    parameters of each random variable x[idx].

    Parameters
    ----------
    shape : int or tuple, optional
        Shape of random variable(s). For multivariate distributions,
        the outermost dimension denotes the multivariate dimension.
    """
    def __init__(self, shape=1):
        get_session()
        if isinstance(shape, int):
            shape = (shape, )

        self.shape = shape
        self.num_vars = np.prod(self.shape)
        self.num_params = None
        self.sample_tensor = False

    def sample_noise(self, size=1):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)

        Returns
        -------
        tf.Tensor
            If size = 1, shape array of type tf.float32. If size > 1,
            (size x shape) array of type tf.float32, where each slice
            along the first dimension is a sample from s.
        """
        raise NotImplementedError()

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)

        Returns
        -------
        tf.Tensor
            If size = 1, shape array of type tf.float32. If size > 1,
            (size x shape) array of type tf.float32, where each slice
            along the first dimension is a sample from s.
        """
        raise NotImplementedError()

    def sample(self, size=1):
        """
        x ~ p(x | params)

        Returns
        -------
        tf.Tensor or np.ndarray
            If size = 1, shape array of type tf.float32 or np.float32.
            If size > 1, (size x shape) array of type tf.float32, where
            each slice along the first dimension is a sample from p.

        Notes
        -----
        If the flag sample_tensor is true, the return object is a
        TensorFlow tensor. Otherwise the return object is a
        realization of a TensorFlow tensor, i.e., NumPy array. The
        latter is required when we require NumPy/SciPy in order to
        sample from distributions.

        The method defaults to sampling noise and reparameterizing it
        (an error is raised if this is not possible).
        """
        return self.reparam(self.sample_noise(size))

    def log_prob(self, xs):
        """
        log p(xs | params)

        Parameters
        ----------
        xs : tf.Tensor or np.ndarray
            (n_minibatch x self.shape) or self.shape

        Returns
        -------
        tf.Tensor
            If xs has dimensions (n_minibatch x self.shape), a vector
            [ sum_{idx in shape} log p(xs[1, idx] | params[idx]), ...,
              sum_{idx in shape} log p(xs[S, idx] | params[idx]) ]
            If xs has dimensions self.shape, a scalar
            sum_{idx in shape} log p(xs[idx] | params[idx])
        """
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1:
            n_minibatch = xs.get_shape()[0]
            log_prob = tf.zeros([n_minibatch], dtype=tf.float32)
        else:
            log_prob = tf.constant(0.0)

        # TODO loop over all possible indices
        #for l in range(len(self.shape)):
        # Loop over each random variable.
        for i in range(self.shape[0]):
            idx = (i, )
            log_prob += self.log_prob_idx(idx, xs)

        return log_prob

    def log_prob_idx(self, idx, xs):
        """
        log p(xs[:, idx] | params[idx])

        Parameters
        ----------
        idx : tuple
            Index of the random variable to take the log density of.
            If univariate distribution, idx is of length
            len(self.shape). If multivariate distribution, idx is of
            length len(self.shape[:-1]).
        xs : tf.Tensor or np.ndarray
            (n_minibatch x self.shape) or self.shape

        Returns
        -------
        tf.Tensor
            If xs has dimensions (n_minibatch x self.shape), a vector
            [ log p(xs[1, idx] | params[idx]), ...,
              log p(xs[S, idx] | params[idx]) ]
            If xs has dimensions self.shape, a scalar
            log p(xs[idx] | params[idx]).
        """
        raise NotImplementedError()

    def entropy(self):
        """
        H(p(x| params))
        = E_{p(x | params)} [ - log p(x | params) ]
        = sum_{idx in shape}
          E_{p(x[idx] | params[idx])} [ - log p(x[idx] | params[idx]) ]

        Returns
        -------
        tf.Tensor
            scalar
        """
        raise NotImplementedError()

class Bernoulli(Distribution):
    """
    p(x | params) = prod_{idx in shape} Bernoulli(x[idx] | p[idx])
    where params = p.
    """
    def __init__(self, shape=1, p=None):
        Distribution.__init__(self, shape)
        self.num_params = self.num_vars
        self.sample_tensor = False

        if p is None:
            p_unconst = tf.Variable(tf.random_normal(self.shape))
            p = tf.sigmoid(p_unconst)

        self.p = p

    def __str__(self):
        p = self.p.eval()
        return "probability: \n" + p.__str__()

    def sample(self, size=1):
        """x ~ p(x | params)"""
        p = self.p.eval()
        return bernoulli.rvs(p, size=size)

    def log_prob_idx(self, idx, xs):
        """log p(xs[:, idx] | params[idx])"""
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1: # n_minibatch x shape
            full_idx = (slice(0, None), ) + idx
        elif rank == len(self.shape): # shape
            full_idx = idx
        else:
            raise IndexError()

        return bernoulli.logpmf(xs[full_idx], self.p[idx])

    def entropy(self):
        # TODO double check entropy can even handle tensors with the
        # right vectorization
        return tf.reduce_sum(bernoulli.entropy(self.p))

class Beta(Distribution):
    """
    p(x | params) = prod_{idx in shape} Beta(x[idx] | alpha[idx], beta[idx])
    where params = {alpha, beta}.
    """
    def __init__(self, shape=1, alpha=None, beta=None):
        Distribution.__init__(self, shape)
        self.num_params = 2*self.num_vars
        self.sample_tensor = False

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
        """x ~ p(x | params)"""
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return beta.rvs(a, b, size=size)

    def log_prob_idx(self, idx, xs):
        """log p(xs[:, idx] | params[idx])"""
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1: # n_minibatch x shape
            full_idx = (slice(0, None), ) + idx
        elif rank == len(self.shape): # shape
            full_idx = idx
        else:
            raise IndexError()

        return beta.logpdf(xs[full_idx], self.alpha[idx], self.beta[idx])

    def entropy(self):
        return tf.reduce_sum(beta.entropy(self.alpha, self.beta))

class Dirichlet(Distribution):
    """
    p(x | params) = prod_{idx in shape[:-1]} Dirichlet(x[idx] | alpha[idx])

    where x is a flattened vector such that x[idx] represents
    a multivariate random variable, and params = params.
    shape[-1] denotes the multivariate dimension.
    """
    def __init__(self, shape, alpha=None):
        Distribution.__init__(self, shape)
        self.num_params = self.num_vars
        self.sample_tensor = False

        if alpha is None:
            alpha_unconst = tf.Variable(tf.random_normal(self.shape))
            alpha = tf.nn.softplus(alpha_unconst)

        self.alpha = alpha

    def __str__(self):
        alpha = self.alpha.eval()
        return "concentration vector: \n" + alpha.__str__()

    def sample(self, size=1):
        """x ~ p(x | params)"""
        alpha = self.alpha.eval()
        return dirichlet.rvs(alpha, size=size)

    def log_prob_idx(self, idx, xs):
        """
        log p(xs[:, idx, :] | params[idx, :])
        where idx is of dimension shape[:-1]
        """
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1: # n_minibatch x shape
            # TODO can i really handle that?
            full_idx = (slice(0, None), ) + idx + (slice(0, None), )
        elif rank == len(self.shape): # shape
            full_idx = idx + (slice(0, None), )
        else:
            raise IndexError()

        return dirichlet.logpdf(xs[full_idx], self.alpha)

    def entropy(self):
        return tf.reduce_sum(dirichlet.entropy(self.alpha))

class InvGamma(Distribution):
    """
    p(x | params) = prod_{idx in shape} Inv_Gamma(x[idx] | alpha[idx], beta[idx])
    where params = {alpha, beta}.
    """
    def __init__(self, shape=1, alpha=None, beta=None):
        Distribution.__init__(self, shape)
        self.num_params = 2*self.num_vars
        self.sample_tensor = False

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
        """x ~ p(x | params)"""
        sess = get_session()
        a, b = sess.run([self.alpha, self.beta])
        return invgamma.rvs(a, b, size=size)

    def log_prob_idx(self, idx, xs):
        """log p(xs[:, idx] | params[idx])"""
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1: # n_minibatch x shape
            full_idx = (slice(0, None), ) + idx
        elif rank == len(self.shape): # shape
            full_idx = idx
        else:
            raise IndexError()

        return invgamma.logpdf(xs[full_idx], self.alpha[idx], self.beta[idx])

    def entropy(self):
        return tf.reduce_sum(invgamma.entropy(self.alpha, self.beta))

class Multinomial(Distribution):
    """
    p(x | params ) = prod_{idx in shape[:-1]} Multinomial(x[idx] | pi[idx])

    where x is a flattened vector such that x[idx] represents
    a multivariate random variable, and params = pi.
    shape[-1] denotes the multivariate dimension.

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
        shape_minus = self.shape[:-1]
        K_minus_one = self.shape[-1] - 1
        self.num_params = np.prod(shape_minus) * K_minus_one
        self.sample_tensor = False

        if pi is None:
            # Transform a real tensor with outer dimension (K-1) to
            # simplex tensor with outer dimension K.
            pi_unconst = tf.Variable(tf.random_normal([real_shape, K_minus_one]))
            logs = -tf.log(tf.cast(K_minus_one - tf.range(K_minus_one),
                                   dtype=tf.float32))
            # TODO dimensions
            x = tf.sigmoid(pi_unconst + logs)
            pil = tf.concat(1, [x, tf.ones([real_shape, 1])])
            piu = tf.concat(1, [tf.ones([real_shape, 1]), 1.0 - x])
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
        return multinomial.rvs(1, pi, size=size)

    def log_prob_idx(self, idx, xs):
        """
        log p(xs[:, idx, :] | params[idx, :])
        where idx is of dimension shape[:-1]
        """
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1: # n_minibatch x shape
            # TODO can i really handle that?
            full_idx = (slice(0, None), ) + idx + (slice(0, None), )
        elif rank == len(self.shape): # shape
            full_idx = idx + (slice(0, None), )
        else:
            raise IndexError()

        return multinomial.logpmf(xs[full_idx], 1, self.pi)

    def entropy(self):
        return tf.reduce_sum(multinomial.entropy(1, self.pi))

class Normal(Distribution):
    """
    p(x | params ) = prod_{idx in shape} Normal(x[idx] | loc[idx], scale[idx])
    where params = {loc, scale}.
    """
    def __init__(self, shape=1, loc=None, scale=None):
        Distribution.__init__(self, shape)
        self.num_params = 2*self.num_vars
        self.sample_tensor = True

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
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)
        """
        if size == 1:
            return tf.random_normal(self.shape)
        else:
            return tf.random_normal((size, ) + self.shape)

    def reparam(self, eps):
        """
        eps = sample_noise() ~ s(eps)
        s.t. x = reparam(eps; params) ~ p(x | params)
        """
        return self.loc + eps * self.scale

    def log_prob_idx(self, idx, xs):
        """log p(xs[:, idx] | params[idx])"""
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1: # n_minibatch x shape
            full_idx = (slice(0, None), ) + idx
        elif rank == len(self.shape): # shape
            full_idx = idx
        else:
            raise IndexError()

        return norm.logpdf(xs[full_idx], self.loc[idx], self.scale[idx])

    def entropy(self):
        return tf.reduce_sum(norm.entropy(scale=self.scale))

class PointMass(Distribution):
    """
    Point mass distribution

    p(x | params ) = prod_{idx in shape} Dirac(x[idx] | params[idx])

    Dirac(x; p) is the Dirac delta distribution with density equal to
    1 if x == p and 0 otherwise.
    """
    def __init__(self, shape=1, params=None):
        Distribution.__init__(self, shape)
        self.num_params = self.num_vars
        self.sample_tensor = True

        if params is None:
            params = tf.Variable(tf.random_normal(self.shape))

        self.params = params

    def __str__(self):
        if self.params.get_shape()[0] == 0:
            return "parameter values: \n" + "None"

        params = self.params.eval()
        return "parameter values: \n" + params.__str__()

    def sample(self, size=1):
        """
        Return a tensor where slices along the first dimension is
        the same set of parameters. This is to be compatible with
        probability model methods which assume the input is possibly
        a batch of parameter samples (as in black box variational
        methods).
        """
        return tf.pack([self.params]*size)

    def log_prob_idx(self, idx, xs):
        """
        log p(xs[:, idx] | params[idx])

        Returns
        -------
        If xs has dimensions n_minibatch x self.shape, a vector where
        the jth element is 1 if xs[j, idx] is equal to params[idx], 0
        otherwise. If xs has dimensions self.shape, a scalar.
        """
        if isinstance(xs, tf.Tensor):
            rank = len(xs.get_shape())
        else: # NumPy array
            rank = len(xs.shape)

        if rank == len(self.shape) + 1: # n_minibatch x shape
            full_idx = (slice(0, None), ) + idx
        elif rank == len(self.shape): # shape
            full_idx = idx
        else:
            raise IndexError()

        return tf.cast(tf.equal(xs[full_idx], self.params[idx]), dtype=tf.float32)
