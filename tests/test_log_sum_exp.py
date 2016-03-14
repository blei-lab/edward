from __future__ import print_function
import unittest
import tensorflow as tf
import numpy as np

from blackbox.util import log_sum_exp

class test_log_sum_exp(unittest.TestCase):

  def test_accuracy(self):
    sess = tf.InteractiveSession()

    x = tf.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
    result = log_sum_exp(x)

    hand_derived_result = -0.5598103014388045

    self.assertAlmostEqual(result.eval(), hand_derived_result)

if __name__ == '__main__':
    unittest.main()
