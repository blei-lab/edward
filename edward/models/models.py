from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import get_dims, get_session
from edward.models.distributions import Normal

try:
    import pystan
    from collections import OrderedDict
except ImportError:
    pass

try:
    import pymc3 as pm
except ImportError:
    pass


class PyMC3Model(object):
    """Model wrapper for models written in PyMC3.
    """
    def __init__(self, model, observed):
        """
        Parameters
        ----------
        model : pymc3.Model
            The probability model
        observed : Theano tensor
            The shared theano tensor passed to the model likelihood
        """
        self.model = model
        self.observed = observed

        vars = pm.inputvars(model.cont_vars)

        bij = pm.DictToArrayBijection(pm.ArrayOrdering(vars), model.test_point)
        self.logp = bij.mapf(model.fastlogp)
        self.dlogp = bij.mapf(model.fastdlogp(vars))

        self.num_vars = len(vars)

    def log_prob(self, xs, zs):
        return tf.py_func(self._py_log_prob, [xs, zs], [tf.float32])[0]

    def _py_log_prob(self, xs, zs):
        n_minibatch = zs.shape[0]
        lp = np.zeros(n_minibatch, dtype=np.float32)
        self.observed.set_value(xs)
        for s in range(n_minibatch):
            lp[s] = self.logp(zs[s, :])

        return lp


class PythonModel(object):
    """Model wrapper for models written in NumPy/SciPy.
    """
    def __init__(self):
        self.num_vars = None

    def log_prob(self, xs, zs):
        """
        Parameters
        ----------
        xs : np.ndarray

        zs : np.ndarray
            n_minibatch x dim(z) array, where each row is a set of
            latent variables.

        Returns
        -------
        tf.py_func
            a TensorFlow op wrapped as a Python function that returns
            n_minibatch array of type np.float32, where each element
            is the log pdf evaluated at (z_{b1}, ..., z_{bd})
        """
        return tf.py_func(self._py_log_prob, [xs, zs], [tf.float32])[0]

    def _py_log_prob(self, xs, zs):
        raise NotImplementedError()


class StanModel(object):
    """Model wrapper for models written in Stan.
    """
    def __init__(self, file=None, model_code=None):
        """
        Parameters
        ----------
        file: see documentation for argument in pystan.stan
        model_code: see documentation for argument in pystan.stan
        """
        if file is not None:
            self.file =  file
        elif model_code is not None:
            self.model_code = model_code
        else:
            raise

        self.flag_init = False

    def log_prob(self, xs, zs):
        if self.flag_init is False:
            self._initialize(xs)

        return tf.py_func(self._py_log_prob, [zs], [tf.float32])[0]

    def _initialize(self, xs):
        print("The following message exists as Stan instantiates the model.")
        if hasattr(self, 'file'):
            self.model = pystan.stan(file=self.file,
                                     data=xs, iter=1, chains=1)
        else:
            self.model = pystan.stan(model_code=self.model_code,
                                     data=xs, iter=1, chains=1)

        self.num_vars = sum([sum(dim) if sum(dim) != 0 else 1 \
                             for dim in self.model.par_dims])
        self.flag_init = True

    def _py_log_prob(self, zs):
        """
        Notes
        -----
        The log_prob() method in Stan requires the input to be a
        dictionary data type, with each parameter named
        correspondingly; this is because zs lives on the original
        (constrained) latent variable space.

        Ideally, in Stan it would have log_prob() for both this
        input and a flattened vector. Internally, Stan always assumes
        unconstrained parameters are flattened vectors, and
        constrained parameters are named data structures.
        """
        lp = np.zeros((zs.shape[0]), dtype=np.float32)
        for b, z in enumerate(zs):
            z_dict = OrderedDict()
            idx = 0
            for dim, par in zip(self.model.par_dims, self.model.model_pars):
                elems = np.sum(dim)
                if elems == 0:
                    z_dict[par] = float(z[idx])
                    idx += 1
                else:
                    z_dict[par] = z[idx:(idx+elems)].reshape(dim)
                    idx += elems

            z_unconst = self.model.unconstrain_pars(z_dict)
            lp[b] = self.model.log_prob(z_unconst, adjust_transform=False)

        return lp


class Variational(object):
    """A container for collecting distribution objects."""
    def __init__(self, layers=None):
        get_session()
        if layers is None:
            self.layers = []
            self.shape = []
            self.num_vars = 0
            self.num_params = 0
            self.is_reparam = True
            self.is_normal = True
            self.is_entropy = True
            self.sample_tensor = []
            self.is_multivariate = []
        else:
            self.layers = layers
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
            self.is_multivariate = [layer.is_multivariate for layer in self.layers]

    def __str__(self):
        string = ""
        for l, layer in enumerate(self.layers):
            if l != 0:
                string += "\n"

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
        self.is_multivariate += [layer.is_multivariate]

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
            If more than one layer, a list of tf.Tensors of dimension
            (size x shape), one for each layer. If one layer, a
            tf.Tensor of (size x shape). If a layer requires SciPy to
            sample, its corresponding tensor is a tf.placeholder.
        """
        samples = []
        for layer in self.layers:
            if layer.sample_tensor:
                samples += [layer.sample(size)]
            else:
                samples += [tf.placeholder(tf.float32, (size, ) + layer.shape)]

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
            If more than one layer, a list of tf.Tensors of dimension
            (batch x shape). If one layer, a tf.Tensor of (batch x
            shape).

        Notes
        -----
        This method assumes each samples[l] in samples has the same
        batch size, i.e., dimensions (batch x shape) for fixed batch
        and varying shape.
        """
        if not isinstance(samples, list):
            samples = [samples]

        size = get_dims(samples[0])[0]
        feed_dict = {}
        for sample, layer in zip(samples, self.layers):
            if sample.name.startswith('Placeholder'):
                feed_dict[sample] = layer.sample(size)

        return feed_dict

    def log_prob(self, xs):
        """
        Parameters
        ----------
        xs : list or tf.Tensor or np.array
            If more than one layer, a list of tf.Tensors or np.array's
            of dimension (batch x shape). If one layer, a tf.Tensor or
            np.array of (batch x shape).

        Notes
        -----
        This method may be removed in the future in favor of indexable
        log_prob methods, e.g., for automatic Rao-Blackwellization.

        This method assumes each xs[l] in xs has the same batch size,
        i.e., dimensions (batch x shape) for fixed batch and varying
        shape.

        This method assumes length of xs == length of self.layers.
        """
        if len(self.layers) == 1:
            return self.layers[0].log_prob(xs)

        n_minibatch = get_dims(xs[0])[0]
        log_prob = tf.zeros([n_minibatch], dtype=tf.float32)
        for l, layer in enumerate(self.layers):
            log_prob += layer.log_prob(xs[l])

        return log_prob

    def entropy(self):
        out = tf.constant(0.0, dtype=tf.float32)
        for layer in self.layers:
            out += layer.entropy()

        return out
