from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multinomial
from scipy import stats

sess = tf.Session()


def _test(n, p, size):
    val_est = multinomial.rvs(n, p, size=size).shape
    val_true = (size, ) + np.asarray(p).shape
    assert val_est == val_true


def test_1d():
    _test(3, np.array([0.4, 0.6]), 1)
    _test(np.array(3), np.array([0.4, 0.6]), 5)


#def test_2d():
#    _test(np.array([3]), np.array([[0.4, 0.6]]), 1)
#    _test(np.array([3]), np.array([[0.4, 0.6]]), 5)
#    _test(np.array([3, 2]), np.array([[0.2, 0.8], [0.6, 0.4]]), 1)
#    _test(np.array([3, 2]), np.array([[0.2, 0.8], [0.6, 0.4]]), 10)
