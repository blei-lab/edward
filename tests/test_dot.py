from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import dot

sess = tf.Session()

a = tf.ones([5]) * np.arange(5)
b = tf.diag(tf.ones([5]))


def test_dot():
    with sess.as_default():
        assert np.all(dot(a, b).eval() == a.eval()[np.newaxis].dot(b.eval()))
        assert np.all(dot(b, a).eval() == b.eval().dot(a.eval()[:, np.newaxis]))
