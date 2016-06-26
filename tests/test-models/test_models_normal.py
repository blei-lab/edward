from __future__ import print_function
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Normal
from scipy import stats

sess = tf.Session()
ed.set_seed(98765)

def _test_log_prob_i(n_minibatch, num_factors):
    normal = Normal(num_factors,
                    loc=tf.constant([0.0] * num_factors),
                    scale=tf.constant([1.0] * num_factors))
    with sess.as_default():
        m = normal.m.eval()
        s = normal.s.eval()
        z = np.random.randn(n_minibatch, num_factors)
        for i in range(num_factors):
            assert np.allclose(
                normal.log_prob_i(i, tf.constant(z, dtype=tf.float32)).eval(),
                stats.norm.logpdf(z[:, i], m[i], s[i]))

def test_log_prob_i_1d_1v():
    _test_log_prob_i(1, 1)

def test_log_prob_i_2d_1v():
    _test_log_prob_i(2, 1)

def test_log_prob_i_1d_2v():
    _test_log_prob_i(1, 2)

def test_log_prob_i_2d_2v():
    _test_log_prob_i(2, 2)
