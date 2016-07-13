from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

sess = tf.Session()


def _next_onepass(data, n_data, N=10):
    if n_data is None:
        n_data = N

    samples = []
    for _ in range(int(N / n_data)):
        # NOTE: the test will only work if N % n_data == 0
        samples.append(data.next(n_data))

    return samples


def _test(data, n_data):
    samples1 = _next_onepass(data, n_data)
    samples2 = _next_onepass(data, n_data)
    for (s1, s2) in zip(samples1, samples2):
        assert np.all(s1 == s2)


def test_single_next():
    data = ed.DataGenerator(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]))
    _test(data, 1)


def test_multiple_next():
    data = ed.DataGenerator(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]))
    _test(data, 2)


def test_all_next():
    data = ed.DataGenerator(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]))
    _test(data, None)
