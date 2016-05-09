from __future__ import print_function
import numpy as np
import tensorflow as tf

try:
    import pystan
    from collections import OrderedDict
except ImportError:
    pass

try:
    import pymc3 as pm
except ImportError:
    pass


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
