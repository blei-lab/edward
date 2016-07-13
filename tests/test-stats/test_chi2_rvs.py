from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import chi2
from scipy import stats

sess = tf.Session()


def _test(df, size):
    val_est = chi2.rvs(df, size=size).shape
    val_true = (size, ) + np.asarray(df).shape
    assert val_est == val_true


def test_0d():
    _test(3, 1)
    _test(np.array(3), 1)


def test_1d():
    _test(np.array([3]), 1)
    _test(np.array([3]), 5)
    _test(np.array([3, 2]), 1)
    _test(np.array([3, 2]), 10)


#def test_2d():
#    _test(np.array([[3]]), 1)
#    _test(np.array([[3]]), 5)
#    _test(np.array([[3, 2]]), 1)
#    _test(np.array([[3, 2]]), 10)
#    _test(np.array([[3, 2], [7, 4]]), 1)
#    _test(np.array([[3, 2], [7, 4]]), 10)
