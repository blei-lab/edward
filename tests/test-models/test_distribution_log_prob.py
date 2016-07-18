from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli
from scipy import stats

sess = tf.Session()
ed.set_seed(98765)


def _test(shape, n):
    # using Bernoulli's internally implemented log_prob_idx() to check
    # Distribution's log_prob()
    rv = Bernoulli(shape, p=tf.zeros(shape)+0.5)
    rv_sample = rv.sample(n)
    with sess.as_default():
        x = rv_sample.eval()
        x_tf = tf.constant(x, dtype=tf.float32)
        p = rv.p.eval()
        val_ed = rv.log_prob(x_tf).eval()
        val_true = 0.0
        for idx in range(shape[0]):
            val_true += stats.bernoulli.logpmf(x[:, idx], p[idx])

        assert np.allclose(val_ed, val_true)


def test_1d():
    _test((1, ), 1)
    _test((1, ), 5)
    _test((5, ), 1)
    _test((5, ), 5)
