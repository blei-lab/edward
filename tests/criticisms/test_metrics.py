from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.criticisms.evaluate import *

all_classification_metrics = [
    binary_accuracy,
    sparse_categorical_accuracy,
]

all_real_classification_metrics = [
    binary_crossentropy,
    categorical_crossentropy,
    hinge,
    squared_hinge,
]

all_regression_metrics = [
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    poisson,
    cosine_proximity,
]

all_specialized_input_output_metrics = [
    categorical_accuracy,
    sparse_categorical_crossentropy,
]


class test_metrics_class(tf.test.TestCase):

  def test_classification_metrics(self):
    with self.test_session():
      y_true = tf.convert_to_tensor(np.random.randint(0, 1, (2, 3)))
      y_pred = tf.convert_to_tensor(np.random.randint(0, 1, (2, 3)))
      for metric in all_classification_metrics:
        self.assertEqual(metric(y_true, y_pred).eval().shape, ())

  def test_real_classification_metrics(self):
    with self.test_session():
      y_true = tf.convert_to_tensor(np.random.randint(0, 5, (6, 7)))
      y_pred = tf.random_normal([6, 7])
      for metric in all_real_classification_metrics:
        self.assertEqual(metric(y_true, y_pred).eval().shape, ())

  def test_regression_metrics(self):
    with self.test_session():
      y_true = tf.random_normal([6, 7])
      y_pred = tf.random_normal([6, 7])
      for metric in all_regression_metrics:
        self.assertEqual(metric(y_true, y_pred).eval().shape, ())

  def test_specialized_input_output_metrics(self):
    with self.test_session():
      for metric in all_specialized_input_output_metrics:
        if metric == categorical_accuracy:
          y_true = tf.convert_to_tensor(np.random.randint(0, 1, (6, 7)))
          y_pred = tf.convert_to_tensor(np.random.randint(0, 7, (6,)))
          self.assertEqual(metric(y_true, y_pred).eval().shape, ())
        elif metric == sparse_categorical_crossentropy:
          y_true = tf.convert_to_tensor(np.random.randint(0, 5, (6)))
          y_pred = tf.random_normal([6, 7])
          self.assertEqual(metric(y_true, y_pred).eval().shape, ())
        else:
          raise NotImplementedError()

if __name__ == '__main__':
  tf.test.main()
