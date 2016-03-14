from __future__ import print_function
import blackbox as bb
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

data = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]


def _sample_onepass(data, n_samples):
    samples = []
    for _ in xrange(data.N / n_samples):
        # NOTE: the test will only work if data.N % n_samples == 0
        samples.append(data.sample(n_data=n_samples))
    return samples


def _assert_eq_tf(samples1, samples2):
    for (s1, s2) in zip(samples1, samples2):
        assert np.all(tf.equal(s1, s2).eval())


def _assert_eq_ndarray(samples1, samples2):
    for (s1, s2) in zip(samples1, samples2):
        assert np.all(s1 == s2)


def _test(data, n_samples, _eq):
    samples1 = _sample_onepass(data, n_samples)
    samples2 = _sample_onepass(data, n_samples)
    _eq(samples1, samples2)


def test_tf_single_sample():
    data_tf = bb.Data(tf.constant(data, dtype=tf.float32),
                      shuffled=True)
    _test(data_tf, 1, _assert_eq_tf)


def test_tf_multiple_samples():
    data_tf = bb.Data(tf.constant(data, dtype=tf.float32),
                      shuffled=True)
    _test(data_tf, 2, _assert_eq_tf)


def test_ndarray_single_sample():
    data_ndarray = bb.Data(np.array(data))
    _test(data_ndarray, 1, _assert_eq_ndarray)


def test_ndarray_multiple_samples():
    data_ndarray = bb.Data(np.array(data))
    _test(data_ndarray, 2, _assert_eq_ndarray)


def test_dict_single_sample():
    data_dict = bb.Data(dict(N=len(data), y=data))
    _test(data_dict, 1, _assert_eq_ndarray)


def test_dict_multiple_samples():
    data_dict = bb.Data(dict(N=len(data), y=data))
    _test(data_dict, 2, _assert_eq_ndarray)
