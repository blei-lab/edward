from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.stats import multivariate_normal
from scipy import stats

sess = tf.Session()


def _assert_eq(val_ed, val_true):
    with sess.as_default():
        assert np.allclose(val_ed.eval(), val_true)


def test_empty():
    _assert_eq(multivariate_normal.entropy(),
               stats.multivariate_normal.entropy())


def test_1d():
    diag = [1.0, 1.0]
    cov = tf.constant(diag)
    _assert_eq(multivariate_normal.entropy(cov=cov),
               stats.multivariate_normal.entropy(cov=np.diag(diag)))
    _assert_eq(multivariate_normal.entropy(cov=np.diag(diag)),
               stats.multivariate_normal.entropy(cov=np.diag(diag)))


def test_2d_diag():
    cm = [[1.0, 0.0], [0.0, 1.0]]
    cov = tf.constant(cm)
    _assert_eq(multivariate_normal.entropy(cov=cov),
               stats.multivariate_normal.entropy(cov=np.array(cm)))
    _assert_eq(multivariate_normal.entropy(cov=np.array(cm)),
               stats.multivariate_normal.entropy(cov=np.array(cm)))


def test_2d_full():
    cm = [[1.0, 0.9], [0.9, 1.0]]
    cov = tf.constant(cm)
    _assert_eq(multivariate_normal.entropy(cov=cov),
               stats.multivariate_normal.entropy(cov=np.array(cm)))
    _assert_eq(multivariate_normal.entropy(cov=np.array(cm)),
               stats.multivariate_normal.entropy(cov=np.array(cm)))
