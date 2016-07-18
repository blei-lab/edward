from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.util import get_dims, get_session
from edward.models.random_variables import Normal

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
    def __init__(self, model):
        """
        Parameters
        ----------
        model : pymc3.Model
            The probability model, written with Theano shared
            variables to form any observations. The Theano shared
            variables are set during inference.
        """
        self.model = model

        vars = pm.inputvars(model.cont_vars)
        self.n_vars = len(vars)

        bij = pm.DictToArrayBijection(pm.ArrayOrdering(vars), model.test_point)
        self.logp = bij.mapf(model.fastlogp)
        self.dlogp = bij.mapf(model.fastdlogp(vars))

    def log_prob(self, xs, zs):
        """
        Parameters
        ----------
        xs : dict
            Data dictionary. Each key is a data placeholder (Theano
            shared variable) in the PyMC3 model, and its value is the
            corresponding realization (np.ndarray or tf.Tensor).
        zs : list or tf.Tensor
            A list of tf.Tensor's if multiple varational families,
            otherwise a tf.Tensor if single variational family.

        Returns
        -------
        tf.Tensor
            S-vector of type tf.float32,
            [log p(xs, zs[1,:]), .., log p(xs, zs[S,:])].

        Notes
        -----
        It wraps around a Python function. The Python function takes
        inputs of type np.ndarray and outputs a np.ndarray.
        """
        # Store `xs.keys()` so that `_py_log_prob_args` knows how each
        # data value corresponds to a key.
        self.keys = list(six.iterkeys(xs))
        if not xs:
            # If `xs` is an empty dictionary, then store their (empty)
            # values to pass into `_py_log_prob_args`.
            self.values = list(six.itervalues(xs))
            inp = [zs]
        elif isinstance(list(six.itervalues(xs))[0], np.ndarray):
            # If `xs` is a dictionary of NumPy arrays, then store
            # their values to pass into `_py_log_prob_args`.
            self.values = list(six.itervalues(xs))
            inp = [zs]
        else:
            # If `xs` is a dictionary of TensorFlow tensors, then
            # pass the tensors into tf.py_func.
            inp = [zs] + list(six.itervalues(xs))

        return tf.py_func(self._py_log_prob_args, inp, [tf.float32])[0]

    def _py_log_prob_args(self, zs, *args):
        # Set `values` to NumPy arrays that were passed in via
        # `self.values` or via `*args`.
        if hasattr(self, 'values'):
            values = self.values
        else:
            values = args

        # Set data placeholders in PyMC3 model (Theano shared
        # variable) to their realizations (NumPy array).
        for key, value in zip(self.keys, values):
            key.set_value(value)

        n_samples = zs.shape[0]
        lp = np.zeros(n_samples, dtype=np.float32)
        for s in range(n_samples):
            lp[s] = self.logp(zs[s, :])

        return lp


class PythonModel(object):
    """Model wrapper for models written in NumPy/SciPy.
    """
    def __init__(self):
        self.n_vars = None

    def log_prob(self, xs, zs):
        """
        Parameters
        ----------
        xs : dict
            Data dictionary. Each key names a data structure used in
            the model (str), and its value is the corresponding
            corresponding realization (np.ndarray or tf.Tensor).
        zs : list or tf.Tensor
            A list of tf.Tensor's if multiple varational families,
            otherwise a tf.Tensor if single variational family.

        Returns
        -------
        tf.Tensor
            S-vector of type tf.float32,
            [log p(xs, zs[1,:]), .., log p(xs, zs[S,:])].

        Notes
        -----
        It wraps around a Python function. The Python function takes
        inputs of type np.ndarray and outputs a np.ndarray.
        """
        # Store `xs.keys()` so that `_py_log_prob_args` knows how each
        # data value corresponds to a key.
        self.keys = list(six.iterkeys(xs))
        if not xs:
            # If `xs` is an empty dictionary, then store their (empty)
            # values to pass into `_py_log_prob_args`.
            self.values = list(six.itervalues(xs))
            inp = [zs]
        elif isinstance(list(six.itervalues(xs))[0], np.ndarray):
            # If `xs` is a dictionary of NumPy arrays, then store
            # their values to pass into `_py_log_prob_args`.
            self.values = list(six.itervalues(xs))
            inp = [zs]
        else:
            # If `xs` is a dictionary of TensorFlow tensors, then
            # pass the tensors into tf.py_func.
            inp = [zs] + list(six.itervalues(xs))

        return tf.py_func(self._py_log_prob_args, inp, [tf.float32])[0]

    def _py_log_prob_args(self, zs, *args):
        # Set `values` to NumPy arrays that were passed in via
        # `self.values` or via `*args`.
        if hasattr(self, 'values'):
            values = self.values
        else:
            values = args

        xs = {key: value for key, value in zip(self.keys, values)}
        return self._py_log_prob(xs, zs)

    def _py_log_prob(self, xs, zs):
        raise NotImplementedError()


class StanModel(object):
    """Model wrapper for models written in Stan.
    """
    def __init__(self, model=None, *args, **kwargs):
        """
        Parameters
        ----------
        model : pystan.StanModel, optional
            An already compiled Stan model. This is useful to avoid
            recompilation of Stan models both within a session (using
            this argument) and across sessions (by loading a pickled
            pystan.StanModel object and passing it in here).
            Alternatively, one can also pickle the ed.StanModel object
            altogether.
        *args
            Passed into pystan.StanModel.
        **kwargs
            Passed into pystan.StanModel.
        """
        if model is None:
            self.model = pystan.StanModel(*args, **kwargs)
        else:
            self.model = model

        self.modelfit = None
        self.is_initialized = False
        self.n_vars = None

    def log_prob(self, xs, zs):
        """
        Parameters
        ----------
        xs : dict
            Data dictionary. Each key and value is according to
            the Stan program's data block. The key type is str; the
            value type is any supported in the data block.
        zs : list or tf.Tensor
            A list of tf.Tensor's if multiple varational families,
            otherwise a tf.Tensor if single variational family.

        Returns
        -------
        tf.Tensor
            S-vector of type tf.float32,
            [log p(xs, zs[1,:]), .., log p(xs, zs[S,:])].

        Notes
        -----
        It wraps around a Python function. The Python function takes
        inputs of type np.ndarray and outputs a np.ndarray.
        """
        print("The empty sampling message exists for accessing Stan's log_prob method.")
        self.modelfit = self.model.sampling(data=xs, iter=1, chains=1)
        if not self.is_initialized:
            self._initialize()

        return tf.py_func(self._py_log_prob, [zs], [tf.float32])[0]

    def _initialize(self):
        self.is_initialized = True
        self.n_vars = sum([sum(dim) if sum(dim) != 0 else 1
                           for dim in self.modelfit.par_dims])

    def _py_log_prob(self, zs):
        """
        Notes
        -----
        The log_prob() method in Stan requires the input to be on
        the unconstrained space. But the zs live on the original
        (constrained) latent variable space. Therefore we must pass zs
        into unconstrain_pars(), which requires the constrained latent
        variables to be of a particular dictionary type.

        Ideally, in Stan it would have log_prob() for direct
        calculation on the constrained latent variables. Internally,
        Stan always assumes unconstrained parameters are flattened
        vectors, and constrained parameters are named data structures.
        This data conversion can be expensive.
        """
        lp = np.zeros((zs.shape[0]), dtype=np.float32)
        for b, z in enumerate(zs):
            z_dict = OrderedDict()
            idx = 0
            for dim, par in zip(self.modelfit.par_dims, self.modelfit.model_pars):
                elems = np.sum(dim)
                if elems == 0:
                    z_dict[par] = float(z[idx])
                    idx += 1
                else:
                    z_dict[par] = z[idx:(idx+elems)].reshape(dim)
                    idx += elems

            z_unconst = self.modelfit.unconstrain_pars(z_dict)
            lp[b] = self.modelfit.log_prob(z_unconst, adjust_transform=False)

        return lp


class Variational(object):
    """A container for collecting distribution objects."""
    def __init__(self, layers=None):
        get_session()
        if layers is None:
            self.layers = []
            self.shape = []
            self.n_vars = 0
            self.n_params = 0
            self.is_reparameterized = True
            self.is_normal = True
            self.is_entropy = True
            self.is_multivariate = []
        else:
            self.layers = layers
            self.shape = [layer.shape for layer in self.layers]
            self.n_vars = sum([layer.n_vars for layer in self.layers])
            self.n_params = sum([layer.n_params for layer in self.layers])
            self.is_reparameterized = all([layer.is_reparameterized
                                           for layer in self.layers])
            self.is_normal = all([isinstance(layer, Normal)
                                  for layer in self.layers])
            self.is_entropy = all(['entropy' in layer.__class__.__dict__
                                   for layer in self.layers])
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
        self.n_vars += layer.n_vars
        self.n_params += layer.n_params
        self.is_reparameterized = self.is_reparameterized and layer.is_reparameterized
        self.is_entropy = self.is_entropy and 'entropy' in layer.__class__.__dict__
        self.is_normal = self.is_normal and isinstance(layer, Normal)
        self.is_multivariate += [layer.is_multivariate]

    def sample(self, n=1):
        """
        Draws a mix of tensors and placeholders, corresponding to
        TensorFlow-based samplers and SciPy-based samplers depending
        on the layer.

        Parameters
        ----------
        n : int, optional
            Number of samples.

        Returns
        -------
        list or tf.Tensor
            If more than one layer, a list of tf.Tensors of dimension
            (n x shape), one for each layer. If one layer, a
            tf.Tensor of (n x shape). If a layer requires SciPy to
            sample, its corresponding tensor is a tf.placeholder.
        """
        samples = [layer.sample(n) for layer in self.layers]
        if len(samples) == 1:
            samples = samples[0]

        return samples

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

        n_samples = get_dims(xs[0])[0]
        log_prob = tf.zeros([n_samples], dtype=tf.float32)
        for l, layer in enumerate(self.layers):
            log_prob += layer.log_prob(xs[l])

        return log_prob

    def entropy(self):
        out = tf.constant(0.0, dtype=tf.float32)
        for layer in self.layers:
            out += layer.entropy()

        return out
