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
    kl_divergence
]

all_metrics_with_binary_averaging = [
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error
]


class test_metrics_class(tf.test.TestCase):

  def _check_averaging(self, metric, y_true, y_pred):
    n_classes = tf.squeeze(tf.shape(y_true)[-1]).eval()
    class_scores = [metric(y_true[i], y_pred[i]) for i in range(n_classes)]

    # No averaging
    no_average = metric(y_true, y_pred, average=None)
    expected_no_average = tf.stack(class_scores)
    self.assertAllEqual(no_average.eval(), expected_no_average.eval())

    # Macro-averaging
    macro_average = metric(y_true, y_pred, average='macro')
    expected_macro_average = tf.reduce_mean(tf.stack(class_scores))
    self.assertAllEqual(macro_average.eval(), expected_macro_average.eval())

    # Micro-averaging
    micro_average = metric(y_true, y_pred, average='micro')
    expected_micro_average = metric(tf.reshape(y_true, [1, -1]),
                                    tf.reshape(y_pred, [1, -1]))
    self.assertAllEqual(micro_average.eval(), expected_micro_average.eval())

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
        elif metric == kl_divergence:
          y_true = tf.nn.softmax(tf.random_normal([6]))
          y_pred = tf.nn.softmax(tf.random_normal([6]))
          self.assertEqual(metric(y_true, y_pred).eval().shape, ())
        else:
          raise NotImplementedError()

  def test_metrics_with_binary_averaging(self):
    with self.test_session():
      y_true = tf.constant([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
      y_pred = tf.constant([[2.0, 4.0, 6.0], [4.0, 6.0, 8.0], [6.0, 8.0, 10.0]])
      for metric in all_metrics_with_binary_averaging:
        self._check_averaging(metric, y_true, y_pred)

if __name__ == '__main__':
  tf.test.main()
