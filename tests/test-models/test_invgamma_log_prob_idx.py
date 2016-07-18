from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import InvGamma
from scipy import stats

sess = tf.Session()
ed.set_seed(98765)


def _test(shape, n):
    rv = InvGamma(shape, alpha=tf.zeros(shape)+0.5, beta=tf.zeros(shape)+0.5)
    rv_sample = rv.sample(n)
    with sess.as_default():
        x = rv_sample.eval()
        x_tf = tf.constant(x, dtype=tf.float32)
        alpha = rv.alpha.eval()
        beta = rv.beta.eval()
        for idx in range(shape[0]):
            assert np.allclose(
                rv.log_prob_idx((idx, ), x_tf).eval(),
                stats.invgamma.logpdf(x[:, idx], alpha[idx], scale=beta[idx]))


def test_1d():
    _test((1, ), 1)
    _test((1, ), 5)
    _test((5, ), 1)
    _test((5, ), 5)
