from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

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
    size = x.shape[0]
    return np.array([multinomial_logpmf(x[i, :], n, p)
                     for i in range(size)])


def _test(shape, n):
    K = shape[-1]
    rv = Multinomial(shape, pi=tf.constant(1.0/K, shape=shape))
    rv_sample = rv.sample(n)
    with sess.as_default():
        x = rv_sample.eval()
        x_tf = tf.constant(x, dtype=tf.float32)
        pi = rv.pi.eval()
        if len(shape) == 1:
            assert np.allclose(
                rv.log_prob_idx((), x_tf).eval(),
                multinomial_logpmf_vec(x[:, :], 1, pi[:]))
        elif len(shape) == 2:
            for i in range(shape[0]):
                assert np.allclose(
                    rv.log_prob_idx((i, ), x_tf).eval(),
                    multinomial_logpmf_vec(x[:, i, :], 1, pi[i, :]))
        else:
            assert False


def test_1d():
    _test((2, ), 1)
    _test((2, ), 2)


def test_2d():
    _test((1, 2), 1)
    _test((1, 3), 1)
    _test((1, 2), 2)
    _test((2, 2), 1)
    _test((2, 2), 2)
