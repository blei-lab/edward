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
    mu: None, Tensor scalar, vector
    mu: None, Tensor scalar, vector, matrix
    TODO allow minibatch
    """
    d = get_dims(x)[0]
    if mu is None:
        mu = tf.zeros([d])
    elif len(mu.get_shape()) == 0: # scalar
        # TensorFlow can't reverse-mode autodiff tf.fill()
        #mu = tf.fill([d], mu)
        mu = tf.ones([d]) * mu

    if Sigma is None:
        Sigma = tf.diag(tf.ones([d]))
    elif len(Sigma.get_shape()) == 0: # scalar
        #Sigma = tf.diag(tf.fill([1], Sigma))
        Sigma = tf.diag(tf.ones([1]) * Sigma)
    elif len(Sigma.get_shape()) == 1: # vector
        Sigma = tf.diag(Sigma)

    """
    # TensorFlow can't reverse-mode autodiff Cholesky
    L = tf.cholesky(Sigma)
    L_inv = tf.matrix_inverse(L)
    det_Sigma = tf.pow(tf.matrix_determinant(L), 2)
    inner = dot(L_inv, tf.sub(x, mu))
    out = -0.5*d*tf.log(2*np.pi) - \
          0.5*tf.log(det_Sigma) - \
          0.5*tf.matmul(tf.transpose(inner), inner)
    """
    det_Sigma = tf.matrix_determinant(Sigma)
    Sigma_inv = tf.matrix_inverse(Sigma)
    vec = tf.sub(x, mu)
    temp = dot(tf.transpose(vec), Sigma_inv)
    out = -0.5*d*tf.log(2*np.pi) - \
          0.5*tf.log(det_Sigma) - \
          0.5*dot(dot(tf.transpose(vec), Sigma_inv), vec)
    #det_Sigma = Sigma
    #Sigma_inv = 1.0 / Sigma
    #vec = tf.sub(x, mu)
    #temp = dot(tf.transpose(vec), Sigma_inv)
    #out = -0.5*d*tf.log(2*np.pi) - \
    #      0.5*tf.log(det_Sigma) - \
    #      0.5*(vec * Sigma_inv * vec)
    return out

#def gaussian_log_prob(x, mu=None, Sigma=None):
#    """
#    log Gaussian(x; mu, Sigma)

#    Arguments
#    ----------
#    x: Tensor scalar, vector, or matrix
#    """
#    dims = x.get_shape()
#    if len(dims) == 0: # scalar
#        N = 1
#        d = 1
#    elif len(dims) == 1: # vector
#        N = 1
#        d = dims[0].value
#    elif len(dims) == 2: # N x d matrix
#        N = dims[0].value
#        d = dims[1].value
#    else:
#        raise

#    if mu is None:
#        mu = tf.zeros([d])
#    if Sigma is None:
#        Sigma = tf.diag(tf.ones([d]))

#    L = tf.cholesky(Sigma)
#    L_inv = tf.matrix_inverse(L)
#    det_Sigma = tf.pow(tf.matrix_determinant(L), 2)
#    out = tf.zeros([N]) # TODO item assignment
#    for n in range(N):
#        #xn = x[n] # TODO
#        xn = x[n, :]
#        inner = dot(L_inv, tf.sub(xn, mu))
#        out[n] = tf.reshape(
#            -0.5*d*tf.log(2*np.pi) - \
#                0.5*tf.log(det_Sigma) - \
#                0.5*tf.matmul(tf.transpose(inner), inner),
#            [-1])


#    N = 5
#    out = tf.zeros([N])
#    for n in range(N):
#        out[n] = n # some operation happens here, which depends on n

#    return out
