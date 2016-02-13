import numpy as np
import tensorflow as tf

from util import log_beta, dot, get_dims

def bernoulli_log_prob(x, p):
    """
    log Bernoulli(x; p)
    """
    clip_finite = True
    if clip_finite:
        # avoid taking log(0) for float32 inputs
        # TODO: adapt to float64, etc.
        p = tf.clip_by_value(p, 1e-45, 1.0)

    lp = tf.to_float(tf.log(p))
    lp1 = tf.to_float(tf.log(1.0-p))
    return tf.mul(x, lp) + tf.mul(1-x, lp1)

def beta_log_prob(x, alpha=1.0, beta=1.0):
    """
    log Beta(x; alpha, beta)
    """
    log_b = log_beta(alpha, beta)
    return (alpha - 1) * tf.log(x) + (beta-1) * tf.log(1-x) - log_b

def gaussian_log_prob(x, mu=None, Sigma=None):
    """
    log Gaussian(x; mu, Sigma)

    Arguments
    ----------
    x: Tensor scalar, vector
    mu: mean - None, Tensor scalar, vector
    Sigma: variance - None, Tensor scalar, vector, matrix
    TODO allow minibatch
    TODO this doesn't do any error checking
    """
    d = get_dims(x)[0]
    if mu is None:
        r = tf.ones([d]) * x
    elif len(mu.get_shape()) == 0: # scalar
        r = x - mu
    else:
        r = tf.sub(x, mu)

    if Sigma is None:
        Sigma_inv = tf.diag(tf.ones([d]))
        det_Sigma = tf.constant(1.0)
    elif len(Sigma.get_shape()) == 0: # scalar
        Sigma_inv = 1.0 / Sigma
        det_Sigma = Sigma
    elif len(Sigma.get_shape()) == 1: # vector
        Sigma_inv = tf.diag(1.0 / Sigma)
        det_Sigma = tf.reduce_prod(Sigma)
    else:
        Sigma_inv = tf.matrix_inverse(Sigma)
        det_Sigma = tf.matrix_determinant(Sigma)

    if d == 1:
        lps = -0.5*d*tf.log(2*np.pi) - \
              0.5*tf.log(det_Sigma) - \
              0.5*r * Sigma_inv * r
    else:
        lps = -0.5*d*tf.log(2*np.pi) - \
              0.5*tf.log(det_Sigma) - \
              0.5*dot(dot(tf.transpose(r), Sigma_inv), r)
        """
        # TensorFlow can't reverse-mode autodiff Cholesky
        L = tf.cholesky(Sigma)
        L_inv = tf.matrix_inverse(L)
        det_Sigma = tf.pow(tf.matrix_determinant(L), 2)
        inner = dot(L_inv, r)
        out = -0.5*d*tf.log(2*np.pi) - \
              0.5*tf.log(det_Sigma) - \
              0.5*tf.matmul(tf.transpose(inner), inner)
        """
    #lp = tf.reduce_sum(lps)
    #return lps
    return tf.reshape(lps, [-1])
