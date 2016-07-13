from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Multinomial
from edward.util import get_dims

sess = tf.Session()


def _test(shape, p, n):
    x = Multinomial(shape, p)
    val_est = tuple(get_dims(x.sample(n)))
    val_true = (n, ) + shape
    assert val_est == val_true


def test_1d():
    _test((2, ), np.array([0.4, 0.6]), 1)
    _test((2, ), np.array([0.4, 0.6]), 5)
    _test((2, ), tf.constant([0.4, 0.6]), 5)


#def test_2d():
#    _test((1, 2), np.array([[0.4, 0.6]]), 1)
#    _test((1, 2), np.array([[0.4, 0.6]]), 5)
#    _test((2, 2), np.array([[0.2, 0.8], [0.6, 0.4]]), 1)
#    _test((2, 2), np.array([[0.2, 0.8], [0.6, 0.4]]), 10)
