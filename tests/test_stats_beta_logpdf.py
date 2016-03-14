from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import beta
from scipy import stats

sess = tf.Session()


def _assert_eq(val_bb, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no log_beta function, the values here are
        # only an approximation
        assert np.allclose(val_bb.eval(), val_true, atol=1e-4)


def _test_logpdf_scalar(scalar, a=.5, b=.5):
    x = tf.constant(scalar)
    val_true = stats.beta.logpdf(scalar, a, b)
    _assert_eq(beta.logpdf(x, tf.constant(a), tf.constant(b)), val_true)
    _assert_eq(beta.logpdf(x, tf.constant([a]), tf.constant(b)), val_true)
    _assert_eq(beta.logpdf(x, tf.constant(a), tf.constant([b])), val_true)
    _assert_eq(beta.logpdf(x, tf.constant([a]), tf.constant([b])), val_true)


def test_logpdf_scalar():
    _test_logpdf_scalar(0.3)
    _test_logpdf_scalar(0.7)

    _test_logpdf_scalar(.3, a=1., b=1.)
    _test_logpdf_scalar(.7, a=1., b=1.)

    _test_logpdf_scalar(.3, a=.5, b=5.)
    _test_logpdf_scalar(.7, a=.5, b=5.)

    _test_logpdf_scalar(.3, a=5., b=.5)
    _test_logpdf_scalar(.7, a=5., b=.5)


def test_logpdf_1d():
    x = tf.constant([0.5])
    val_true = stats.beta.logpdf([0.5], 0.5, 0.5)
    _assert_eq(beta.logpdf(x, tf.constant(0.5), tf.constant(0.5)), val_true)
    _assert_eq(beta.logpdf(x, tf.constant([0.5]), tf.constant(0.5)), val_true)
    _assert_eq(beta.logpdf(x, tf.constant(0.5), tf.constant([0.5])), val_true)
    _assert_eq(beta.logpdf(x, tf.constant([0.5]), tf.constant([0.5])), val_true)
