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

def _test_logpdf_1d(vector, alpha=np.array([0.5, 0.5])):
    x = tf.constant(vector)
    val_true = stats.dirichlet.logpdf(vector, alpha)
    _assert_eq(dirichlet.logpdf(x, alpha), val_true)
    _assert_eq(dirichlet.logpdf(x, tf.convert_to_tensor(alpha)), val_true)

def test_logpdf_1d():
    _test_logpdf_1d(np.array([0.3, 0.7]))
    _test_logpdf_1d(np.array([0.2, 0.8]))

    _test_logpdf_1d(np.array([0.3, 0.7]), alpha=np.array([1.0, 1.0]))
    _test_logpdf_1d(np.array([0.2, 0.8]), alpha=np.array([1.0, 1.0]))

    _test_logpdf_1d(np.array([0.3, 0.7]), alpha=np.array([0.5, 5.0]))
    _test_logpdf_1d(np.array([0.2, 0.8]), alpha=np.array([0.5, 5.0]))

    _test_logpdf_1d(np.array([0.3, 0.7]), alpha=np.array([5.0, 0.5]))
    _test_logpdf_1d(np.array([0.2, 0.8]), alpha=np.array([5.0, 0.5]))
