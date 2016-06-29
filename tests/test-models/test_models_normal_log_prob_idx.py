from __future__ import print_function
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Normal
from scipy import stats

sess = tf.Session()
ed.set_seed(98765)

def _test(shape, n_minibatch):
    normal = Normal(shape,
                    loc=tf.constant([0.0] * shape),
                    scale=tf.constant([1.0] * shape))
    with sess.as_default():
        m = normal.loc.eval()
        s = normal.scale.eval()
        z = np.random.randn(n_minibatch, shape)
        for i in range(shape):
            assert np.allclose(
                normal.log_prob_idx((i, ), tf.constant(z, dtype=tf.float32)).eval(),
                stats.norm.logpdf(z[:, i], m[i], s[i]))

def test_1v_1d():
    _test(1, 1)

def test_1v_2d():
    _test(1, 2)

def test_2v_1d():
    _test(2, 1)

def test_2v_2d():
    _test(2, 2)
