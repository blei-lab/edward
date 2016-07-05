import edward as ed
import numpy as np
import tensorflow as tf

sess = tf.Session()

def _next_onepass(data, n_data, N=10):
    samples = []
    for _ in range(N / n_data):
        # NOTE: the test will only work if N % n_data == 0
        samples.append(data.next(n_data)['x'])

    return samples

def _assert_eq_tf(samples1, samples2):
    with sess.as_default():
        for (s1, s2) in zip(samples1, samples2):
            assert np.all(tf.equal(s1, s2).eval())

def _assert_eq_ndarray(samples1, samples2):
    for (s1, s2) in zip(samples1, samples2):
        assert np.all(s1 == s2)

def _test(data, n_data, _eq):
    samples1 = _next_onepass(data, n_data)
    samples2 = _next_onepass(data, n_data)
    _eq(samples1, samples2)

def test_single_next():
    data = ed.DataGenerator({'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])})
    _test(data, 1, _assert_eq_ndarray)
    data = ed.DataGenerator({'x': tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])})
    _test(data, 1, _assert_eq_tf)

def test_multiple_next():
    data = ed.DataGenerator({'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])})
    _test(data, 2, _assert_eq_ndarray)
    data = ed.DataGenerator({'x': tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])})
    _test(data, 2, _assert_eq_tf)
