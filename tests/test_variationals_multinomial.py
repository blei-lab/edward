from __future__ import print_function
import edward as ed
import tensorflow as tf
import numpy as np

from edward.variationals import Multinomial
from scipy.special import gammaln

sess = tf.Session()
ed.set_seed(98765)

def multinomial_logpmf(x, n, p):
    """
    Arguments
    ----------
    x: np.array
        vector of length K, where x[i] is the number of outcomes
        in the ith bucket
    n: int
        number of outcomes equal to sum x[i]
    p: np.array
        vector of probabilities summing to 1
    """
    return gammaln(n + 1.0) - \
           np.sum(gammaln(x + 1.0)) + \
           np.sum(x * np.log(p))

def multinomial_logpmf_vec(x, n, p):
    n_minibatch = x.shape[0]
    out = np.zeros(n_minibatch)
    return np.array([multinomial_logpmf(x[i,:], n, p) for i in xrange(n_minibatch)])

def _test_log_prob_zi(n_minibatch, num_factors, K):
    multinomial = Multinomial(num_factors, K)
    multinomial.pi = tf.constant(1.0/K, shape=[num_factors, K])

    with sess.as_default():
        pi = multinomial.pi.eval()
        z = np.zeros((n_minibatch, K*num_factors))
        for i in xrange(num_factors):
            z[:, (i*K):((i+1)*K)] = np.random.multinomial(1, pi[i, :], size=n_minibatch)

        z_tf = tf.constant(z, dtype=tf.float32)
        for i in xrange(num_factors):
            # NOTE: since Tensorflow has no special functions, the values here are
            # only an approximation
            assert np.allclose(
                multinomial.log_prob_zi(i, z_tf).eval(),
                multinomial_logpmf_vec(z[:, (i*K):((i+1)*K)], 1, pi[i, :]),
                atol=1e-4)

def test_log_prob_zi_1d_1v_2k():
    _test_log_prob_zi(1, 1, 2)

def test_log_prob_zi_1d_1v_3k():
    _test_log_prob_zi(1, 1, 3)

def test_log_prob_zi_2d_1v_2k():
    _test_log_prob_zi(2, 1, 2)

def test_log_prob_zi_1d_2v_2k():
    _test_log_prob_zi(1, 2, 2)

def test_log_prob_zi_2d_2v_2k():
    _test_log_prob_zi(2, 2, 2)
