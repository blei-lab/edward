from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward import criticisms

all_classification_metrics = [
    criticisms.binary_accuracy,
    criticisms.categorical_accuracy,
    criticisms.binary_crossentropy,
    criticisms.categorical_crossentropy,
    criticisms.hinge,
    criticisms.squared_hinge,
]

all_sparse_metrics = [
    criticisms.sparse_categorical_accuracy,
    criticisms.sparse_categorical_crossentropy,
]

all_regression_metrics = [
    criticisms.mean_squared_error,
    criticisms.mean_absolute_error,
    criticisms.mean_absolute_percentage_error,
    criticisms.mean_squared_logarithmic_error,
    criticisms.poisson,
    criticisms.cosine_proximity,
]

sess = tf.Session()


def test_classification_metrics():
    y_a = tf.convert_to_tensor(np.random.randint(0, 7, (6, 7)), dtype=tf.float32)
    y_b = tf.convert_to_tensor(np.random.random((6, 7)))
    for metric in all_classification_metrics:
        with sess.as_default():
            assert metric(y_a, y_b).eval().shape == ()


def test_sparse_classification_metrics():
    y_a = tf.convert_to_tensor(np.random.randint(0, 7, (6,)), dtype=tf.float32)
    y_b = tf.convert_to_tensor(np.random.random((6, 7)))
    for metric in all_sparse_metrics:
        with sess.as_default():
            assert metric(y_a, y_b).eval().shape == ()


def test_regression_metrics():
    y_a = tf.convert_to_tensor(np.random.random((6, 7)))
    y_b = tf.convert_to_tensor(np.random.random((6, 7)))
    for metric in all_regression_metrics:
        output = metric(y_a, y_b)
        with sess.as_default():
            assert output.eval().shape == ()
