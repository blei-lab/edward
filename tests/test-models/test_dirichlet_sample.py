from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Dirichlet
from edward.util import get_dims

sess = tf.Session()


def _test(shape, alpha, size):
    x = Dirichlet(shape, alpha)
    val_est = tuple(get_dims(x.sample(size=size)))
    val_true = (size, ) + shape
    assert val_est == val_true


def test_1d():
    _test((2, ), np.array([0.2, 0.8]), 1)
    _test((2, ), np.array([0.2, 0.8]), 10)
    _test((3, ), np.array([0.2, 1.1, 0.8]), 1)
    _test((3, ), np.array([0.2, 1.1, 0.8]), 10)
    _test((2, ), tf.constant([0.2, 0.8]), 1)
    _test((2, ), tf.constant([0.2, 0.8]), 10)
    _test((3, ), tf.constant([0.2, 1.1, 0.8]), 1)
    _test((3, ), tf.constant([0.2, 1.1, 0.8]), 10)


#def test_2d():
#    _test(np.array([[0.2, 0.8]]), 1)
#    _test(np.array([[0.2, 0.8]]), 10)
#    _test(np.array([[0.2, 1.1, 0.8], [0.7, 0.65, 0.6]]), 1)
#    _test(np.array([[0.2, 1.1, 0.8], [0.7, 0.65, 0.6]]), 10)
