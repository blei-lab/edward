from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import chi2
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(x, df):
    xtf = tf.constant(x)
    val_true = stats.chi2.logpdf(x, df)
    _assert_eq(chi2.logpdf(xtf, df), val_true)
    _assert_eq(chi2.logpdf(xtf, tf.constant(df)), val_true)
    _assert_eq(chi2.logpdf(xtf, tf.constant([df])), val_true)


def test_0d():
    _test(0.2, df=2)
    _test(0.623, df=2)


def test_1d():
    _test([0.1, 1.0, 0.58, 2.3], df=3)


def test_2d():
    _test(np.array([[0.1, 1.0, 0.58, 2.3],[0.3, 1.1, 0.68, 1.2]]), df=3)
