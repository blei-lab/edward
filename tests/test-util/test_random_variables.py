from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util.random_variables import compute_multinomial_mode


class test_compute_multinomial_mode(tf.test.TestCase):

    def test_correct_mode_computed_with_uniform_probabilities(self):
      with self.test_session():
        probs = np.array(3 * [1/3.0])
        total_count = 5
        self.assertIn(
          compute_multinomial_mode(probs, total_count),
          [[2, 2, 1], [2, 1, 2], [1, 2, 2]]
        )
        probs = np.array([0.6, 0.4, 0.0])
        total_count = 5
        self.assertEqual(
          compute_multinomial_mode(probs, total_count),
          [3, 2, 0]
        )

if __name__ == '__main__':
  tf.test.main()
