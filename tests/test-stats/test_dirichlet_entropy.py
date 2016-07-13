from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import dirichlet
from scipy import stats

sess = tf.Session()


def dirichlet_entropy_vec(alpha):
    """Vectorized version of stats.dirichlet.entropy."""
    if len(alpha.shape) == 1:
        return stats.dirichlet.entropy(alpha)
    else:
        n_minibatch = alpha.shape[0]
        return np.array([stats.dirichlet.entropy(alpha[i, :])
                         for i in range(n_minibatch)])


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def _test(alpha):
    val_true = dirichlet_entropy_vec(alpha)
    _assert_eq(dirichlet.entropy(alpha), val_true)
    _assert_eq(dirichlet.entropy(tf.convert_to_tensor(alpha)), val_true)


def test_1d():
    _test(alpha=np.array([0.5, 0.5]))
    _test(alpha=np.array([0.5, 0.5]))

    _test(alpha=np.array([1.0, 1.0]))
    _test(alpha=np.array([1.0, 1.0]))

    _test(alpha=np.array([0.5, 5.0]))
    _test(alpha=np.array([0.5, 5.0]))

    _test(alpha=np.array([5.0, 0.5]))
    _test(alpha=np.array([5.0, 0.5]))


def test_2d():
    _test(np.array([[0.3, 0.7],[0.5, 0.5]]))
    _test(np.array([[0.2, 0.8],[0.3, 0.7]]))
