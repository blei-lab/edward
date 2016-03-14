from __future__ import print_function
import numpy as np
import tensorflow as tf
import blackbox as bb

from scipy import stats

sess = tf.Session()
# fix random seed
np.random.seed(98765)


def _test_log_prob_zi(n_data, n_vars):
    variational = bb.MFGaussian(n_vars)
    variational.m_unconst = tf.constant([0.0] * n_vars)
    variational.s_unconst = tf.constant(np.random.randn(n_vars))

    with sess.as_default():
        m = variational.transform_m(variational.m_unconst).eval()
        s = variational.transform_s(variational.s_unconst).eval()

        z = np.random.randn(n_data, n_vars)

        for i in xrange(n_vars):
            assert np.allclose(variational.log_prob_zi(
                i, tf.constant(z, dtype=tf.float32)).eval(),
                stats.norm.logpdf(z[:, i], m[i], s[i]))


def test_log_prob_zi_1d_1v():
    _test_log_prob_zi(1, 1)


def test_log_prob_zi_2d_1v():
    _test_log_prob_zi(2, 1)


def test_log_prob_zi_1d_2v():
    _test_log_prob_zi(1, 2)


def test_log_prob_zi_2d_2v():
    _test_log_prob_zi(2, 2)
