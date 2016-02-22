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

class Model:
    """
    Model wrapper for Stan programs, where
    log p(x, z) = log_prob() method
    nabla_z log p(x, z) = grad_log_prob() method

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

    def log_prob(self, zs):
        return tf.py_func(self._py_log_prob, [zs], [tf.float32])[0]

    # TODO
    #def grad_log_prob(self, zs):
    #    return tf.pack([self.model.grad_log_prob(z) for z in tf.unpack(zs)])

    def _py_log_prob(self, zs):
        return np.array([self.model.log_prob(z) for z in zs], dtype=np.float32)
        # TODO deal with constrain vs unconstrain
        #return np.array([self.model.log_prob(self.model.unconstrain_pars(z)) for z in zs], dtype=np.float32)
