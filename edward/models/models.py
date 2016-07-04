from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.util import get_dims, get_session
from .distributions import Normal

try:
    import pystan
    from collections import OrderedDict
except ImportError:
    pass

try:
    import pymc3 as pm
except ImportError:
    pass

class Model:
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
            Number of samples to draw.

        Returns
        -------
        dict
            Dictionary of distribution objects in the container assigned
            to a tf.Tensor. Each tf.Tensor is of size size x shape.
        """
        samples = {}
        for layer in self.layers:
            if layer.sample_tensor:
                samples[layer] = layer.sample(size)
            else:
                samples[layer] = tf.placeholder(tf.float32, (size, ) + layer.shape)

        return samples

    def np_dict(self, samples):
        """
        Form dictionary to feed any placeholders with np.array
        samples.

        Parameters
        ----------
        samples : dict
            Dictionary of distribution objects in the container assigned
            to a tf.Tensor. Each tf.Tensor is of size batch x shape.

        Return
        ------
        dict
            Dictionary of tf.placeholders in `samples` binded to SciPy
            samples.

        Notes
        -----
        This method assumes each samples[l] in samples has the same
        batch size, i.e., dimensions (batch x shape) for fixed batch
        and varying shape.
        """
        size = get_dims(samples.values()[0])[0]
        feed_dict = {}
        for layer, sample in samples.iteritems():
            if sample.name.startswith('Placeholder'):
                feed_dict[sample] = layer.sample(size)

        # TODO technically this doesn't require anything from self
        return feed_dict

    def log_prob(self, data_dict):
        """
        Parameters
        ----------
        data_dict : dict
            Dictionary which binds all random variables (distribution
            objects) in the model (container object) to realizations
            (tf.Tensor or np.ndarray's). For each random variable of
            dimensions `shape`, its corresponding realization has either
            dimensions `shape` or `batch x shape`. Any optional outer
            dimension `batch` must be the same number for the optional
            outer dimension of each realization.

        Returns
        -------
        tf.Tensor
            If there is an outer dimension batch for at least one
            realization, return object is a vector of batch elements,
            evaluating the log density for each relization in that set of
            realizations and vectorize-summing over the reset. Otherwise a
            scalar.

        Notes
        -----
        This method may be removed in the future in favor of indexable
        log_prob methods, e.g., for automatic Rao-Blackwellization.

        This method assumes length of data_dict == length of self.layers and
        each item corresponds to a layer in self.layers.
        """
        # Get batch size from the first item in the dictionary. For now we
        # assume the outer dimension always has the batch size.
        if isinstance(data_dict.values()[0], tf.Tensor):
            shape = get_dims(data_dict.values()[0])
        else: # NumPy array
            shape = data_dict.values()[0].shape

        # Sum over the log-density of each distribution in container.
        n_minibatch = shape[0]
        log_prob = tf.zeros([n_minibatch], dtype=tf.float32)
        for layer in self.layers:
            log_prob += layer.log_prob(data_dict[layer])

        return log_prob

    def entropy(self):
        out = tf.constant(0.0, dtype=tf.float32)
        for layer in self.layers:
            out += layer.entropy()

        return out

class PyMC3Model:
    """
    Model wrapper for models written in PyMC3.

    Arguments
    ----------
    model : pymc3.Model object
    observed : The shared theano tensor passed to the model likelihood
    """
    def __init__(self, model, observed):
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

class PythonModel:
    """
    Model wrapper for models written in NumPy/SciPy.
    """
    def __init__(self):
        self.num_vars = None

    def log_prob(self, xs, zs):
        return tf.py_func(self._py_log_prob, [xs, zs], [tf.float32])[0]

    def _py_log_prob(self, xs, zs):
        """
        Arguments
        ----------
        xs : np.ndarray

        zs : np.ndarray
            n_minibatch x dim(z) array, where each row is a set of
            latent variables.

        Returns
        -------
        np.ndarray
            n_minibatch array of type np.float32, where each element
            is the log pdf evaluated at (z_{b1}, ..., z_{bd})
        """
        raise NotImplementedError()

class StanModel:
    """
    Model wrapper for models written in Stan.

    Arguments
    ----------
    file: see documentation for argument in pystan.stan
    model_code: see documentation for argument in pystan.stan
    """
    def __init__(self, file=None, model_code=None):
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
