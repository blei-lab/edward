from __future__ import print_function
import numpy as np
import pystan
import tensorflow as tf

def set_seed(x):
    """
    Set seed for both NumPy and TensorFlow.
    """
    np.random.seed(x)
    tf.set_random_seed(x)

def dot(x, y):
    """
    x is M x N matrix and y is N-vector, or
    x is M-vector and y is M x N matrix
    """
    if len(x.get_shape()) == 1:
        vec = x
        mat = y
        d = vec.get_shape()[0].value
        return tf.matmul(tf.reshape(vec, [1, d]), mat)
    else:
        mat = x
        vec = y
        d = vec.get_shape()[0].value
        return tf.matmul(mat, tf.reshape(vec, [d, 1]))

def trace(X):
    # assumes square
    n = X.get_shape()[0].value
    mask = tf.diag(tf.ones([n], dtype=X.dtype))
    X = tf.mul(mask, X)
    return tf.reduce_sum(X)

def get_dims(x):
    """
    Get values of each dimension.

    Arguments
    ----------
    x: tensor scalar or array
    """
    dims = x.get_shape()
    if len(dims) == 0: # scalar
        return [1]
    else: # array
        return [dim.value for dim in dims]

def log_gamma(x):
    """
    TensorFlow doesn't have special functions, so use a
    log/exp/polynomial approximation.
    http://www.machinedlearnings.com/2011/06/faster-lda.html
    """
    logterm = tf.log(x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log (xp3)

def log_beta(x, y):
    """
    TensorFlow doesn't have special functions, so use a
    log/exp/polynomial approximation.
    """
    return log_gamma(x) + log_gamma(y) - log_gamma(x+y)

def logit(x, clip_finite=True):
    if isinstance(x, tf.Tensor):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_logit_input")
        transformed = 1.0 / (1 + tf.exp(-x))
        jacobian = transformed * (1-transformed)
        if clip_finite:
            jacobian = tf.clip_by_value(jacobian, 1e-45, 1e38, name="clipped_jacobian")
        log_jacobian = tf.reduce_sum(tf.log(jacobian))

    else:
        transformed = 1.0 / (1 + np.exp(-x))
        jacobian = transformed * (1-transformed)
        log_jacobian = np.sum(np.log(jacobian))

    return transformed, log_jacobian

class PythonModel:
    """
    Model wrapper for models implemented in NumPy and SciPy.

    Arguments
    ----------
    file: see documentation for argument in pystan.stan
    model_code: see documentation for argument in pystan.stan
    """
    def __init__(self):
        self.num_vars = None

    def log_prob(self, zs):
        return tf.py_func(self._py_log_prob, [zs], [tf.float32])[0]

    # TODO
    #https://github.com/tensorflow/tensorflow/issues/1095
    #https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#RegisterGradient
    # TODO try this first on just raw numpy/scipy, and get it as a mwe
    @tf.RegisterGradient("log_prob") # TODO or is the name self.log_prob?
    def _log_prob_grad(self, zs):
        return tf.py_func(self._py_log_prob_grad, [zs], [tf.float32])[0]

    def _py_log_prob(self, zs):
        """
        Arguments
        ----------
        zs : np.ndarray
            n_minibatch x dim(z) array, where each row is a set of
            latent variables.

        Returns
        -------
        np.ndarray
            n_minibatch array of type np.float32, where each element
            is the log pdf evaluated at (z_{b1}, ..., z_{bd})
        """
        pass

    def _py_log_prob_grad(self, zs):
        """
        Arguments
        ----------
        zs : np.ndarray
            n_minibatch x dim(z) array, where each row is a set of
            latent variables.

        Returns
        -------
        np.ndarray
            n_minibatch x dim(z) array of type np.float32, where each
            row is the gradient of the log pdf with respect to (z_1,
            ..., z_d).
        """
        pass

class StanModel(PythonModel):
    """
    Model wrapper for Stan programs.

    Arguments
    ----------
    file: see documentation for argument in pystan.stan
    model_code: see documentation for argument in pystan.stan
    """
    # TODO pystan should be an optional package unless users want to use this class, and also requirements tensorflow>=0.7.0 for the same reason; otherwise tensorflow>=0.6.0
    def __init__(self, file=None, model_code=None, data=None):
        if data is None:
            raise

        if file is not None:
            print("The following message exists as Stan initializes an empty model.")
            self.model = pystan.stan(file=file,
                                     data=data, iter=1, chains=1)
        elif model_code is not None:
            print("The following message exists as Stan initializes an empty model.")
            self.model = pystan.stan(model_code=model_code,
                                     data=data, iter=1, chains=1)
        else:
            raise

        self.num_vars = len(self.model.par_dims) # TODO

    def _py_log_prob(self, zs):
        return np.array([self.model.log_prob(z) for z in zs], dtype=np.float32)
        # TODO deal with constrain vs unconstrain
        #return np.array([self.model.log_prob(self.model.unconstrain_pars(z)) for z in zs], dtype=np.float32)

    def _py_log_prob_grad(self, zs):
        return np.array([self.model.grad_log_prob(z) for z in zs], dtype=np.float32)
