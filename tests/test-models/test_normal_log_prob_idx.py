from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal
from scipy import stats

sess = tf.Session()
ed.set_seed(98765)


def _test(shape, n):
    rv = Normal(shape, loc=tf.zeros(shape), scale=tf.ones(shape))
    rv_sample = rv.sample(n)
    with sess.as_default():
        x = rv_sample.eval()
        x_tf = tf.constant(x, dtype=tf.float32)
        loc = rv.loc.eval()
        scale = rv.scale.eval()
        for idx in range(shape[0]):
            assert np.allclose(
                rv.log_prob_idx((idx, ), x_tf).eval(),
                stats.norm.logpdf(x[:, idx], loc[idx], scale[idx]))


def test_1d():
    _test((1, ), 1)
    _test((1, ), 5)
    _test((5, ), 1)
    _test((5, ), 5)
