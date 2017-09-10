from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Categorical, Multinomial, Normal


METRICS_WITH_AVERAGING = [
    'binary_accuracy'
]

class test_evaluate_class(tf.test.TestCase):

  def test_metrics_classification_with_binary_averaging(self):
    with self.test_session():
      x = Multinomial(total_count=5.0, probs=tf.constant([0.48, 0.51, 0.01]))
      x_data = tf.constant([3, 2, 0], dtype=x.dtype.as_numpy_dtype)
      self.assertAllClose(
        [1., 1., 1.],
        ed.evaluate(('sparse_categorical_accuracy', {'average': 'micro'}), {x: x_data}, n_samples=1)
      )
