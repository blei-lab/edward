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


class test_metrics_class(tf.test.TestCase):

  def test_classification_metrics(self):
    with self.test_session():
      y_a = tf.convert_to_tensor(np.random.randint(0, 7, (6, 7)),
                                 dtype=tf.float32)
      y_b = tf.convert_to_tensor(np.random.random((6, 7)))
      for metric in all_classification_metrics:
        assert metric(y_a, y_b).eval().shape == ()

  def test_sparse_classification_metrics(self):
    with self.test_session():
      y_a = tf.convert_to_tensor(np.random.randint(0, 7, (6,)),
                                 dtype=tf.float32)
      y_b = tf.convert_to_tensor(np.random.random((6, 7)))
      for metric in all_sparse_metrics:
        assert metric(y_a, y_b).eval().shape == ()

  def test_regression_metrics(self):
    with self.test_session():
      y_a = tf.convert_to_tensor(np.random.random((6, 7)))
      y_b = tf.convert_to_tensor(np.random.random((6, 7)))
      for metric in all_regression_metrics:
        output = metric(y_a, y_b)
        assert output.eval().shape == ()

if __name__ == '__main__':
  tf.test.main()
