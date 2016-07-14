from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Dirichlet
from scipy import stats

sess = tf.Session()
ed.set_seed(98765)


def dirichlet_logpdf_vec(x, alpha):
    """Vectorized version of stats.dirichlet.logpdf."""
    if len(x.shape) == 1:
        return stats.dirichlet.logpdf(x, alpha)
    else:
        size = x.shape[0]
        return np.array([stats.dirichlet.logpdf(x[i, :], alpha)
                         for i in range(size)])


def _test(shape, n):
    K = shape[-1]
    rv = Dirichlet(shape, alpha=tf.constant(1.0/K, shape=shape))
    rv_sample = rv.sample(n)
    with sess.as_default():
        x = rv_sample.eval()
        x_tf = tf.constant(x, dtype=tf.float32)
        alpha = rv.alpha.eval()
        if len(shape) == 1:
            assert np.allclose(
                rv.log_prob_idx((), x_tf).eval(),
                dirichlet_logpdf_vec(x[:, :], alpha[:]))
        elif len(shape) == 2:
            for i in range(shape[0]):
                assert np.allclose(
                    rv.log_prob_idx((i, ), x_tf).eval(),
                    dirichlet_logpdf_vec(x[:, i, :], alpha[i, :]))
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
