from __future__ import print_function
import numpy as np
import tensorflow as tf

from edward.stats import dirichlet
from scipy import stats

sess = tf.Session()

def _assert_eq(val_ed, val_true):
    with sess.as_default():
        # NOTE: since Tensorflow has no special functions, the values here are
        # only an approximation
        assert np.allclose(val_ed.eval(), val_true, atol=1e-4)

def _test_logpdf_1d(x, alpha=np.array([0.5, 0.5])):
    xtf = tf.constant(x)
    val_true = stats.dirichlet.logpdf(x, alpha)
    _assert_eq(dirichlet.logpdf(xtf, alpha), val_true)
    _assert_eq(dirichlet.logpdf(xtf, tf.convert_to_tensor(alpha)), val_true)

def test_logpdf_1d():
    _test_logpdf_1d(np.array([0.3, 0.7]))
    _test_logpdf_1d(np.array([0.2, 0.8]))

    _test_logpdf_1d(np.array([0.3, 0.7]), alpha=np.array([1.0, 1.0]))
    _test_logpdf_1d(np.array([0.2, 0.8]), alpha=np.array([1.0, 1.0]))

    _test_logpdf_1d(np.array([0.3, 0.7]), alpha=np.array([0.5, 5.0]))
    _test_logpdf_1d(np.array([0.2, 0.8]), alpha=np.array([0.5, 5.0]))

    _test_logpdf_1d(np.array([0.3, 0.7]), alpha=np.array([5.0, 0.5]))
    _test_logpdf_1d(np.array([0.2, 0.8]), alpha=np.array([5.0, 0.5]))
