from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Multinomial
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
    n = x.shape[0]
    return np.array([multinomial_logpmf(x[i, :], n, p)
                     for i in range(n)])


def _test(shape, n):
    K = shape[-1]
    multinomial = Multinomial(shape, pi=tf.constant(1.0/K, shape=shape))
    with sess.as_default():
        pi = multinomial.pi.eval()
        z = np.zeros((n, ) + tuple(shape))
        for i in range(shape[0]):
            z[:, i, :] = np.random.multinomial(1, pi[i, :], n=n)

        z_tf = tf.constant(z, dtype=tf.float32)
        for i in range(shape[0]):
            assert np.allclose(
                multinomial.log_prob_idx((i, ), z_tf).eval(),
                multinomial_logpmf_vec(z[:, i, :], 1, pi[i, :]))


def test_1_2v_1d():
    _test([1, 2], 1)


def test_1_3v_1d():
    _test([1, 3], 1)


def test_1_2v_2d():
    _test([1, 2], 2)


def test_2_2v_1d():
    _test([2, 2], 1)


def test_2_2v_2d():
    _test([2, 2], 2)
