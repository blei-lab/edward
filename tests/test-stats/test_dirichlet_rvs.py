from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import dirichlet
from scipy import stats

sess = tf.Session()


def _test(alpha, size):
    val_est = dirichlet.rvs(alpha, size=size).shape
    val_true = (size, ) + np.asarray(alpha).shape
    assert val_est == val_true


def test_1d():
    _test(np.array([0.2, 0.8]), 1)
    _test(np.array([0.2, 0.8]), 10)
    _test(np.array([0.2, 1.1, 0.8]), 1)
    _test(np.array([0.2, 1.1, 0.8]), 10)


#def test_2d():
#    _test(np.array([[0.2, 0.8]]), 1)
#    _test(np.array([[0.2, 0.8]]), 10)
#    _test(np.array([[0.2, 1.1, 0.8], [0.7, 0.65, 0.6]]), 1)
#    _test(np.array([[0.2, 1.1, 0.8], [0.7, 0.65, 0.6]]), 10)
