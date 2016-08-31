from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import logit


class test_logit_class_class(tf.test.TestCase):

  def test_logit_0d(self):
    with self.test_session():
      x = tf.constant(0.01)
      self.assertAllClose(logit(x).eval(),
                          -4.5951198501345898)
      x = tf.constant(0.25)
      self.assertAllClose(logit(x).eval(),
                          -1.0986122886681096)
      x = tf.constant(0.5)
      self.assertAllEqual(logit(x).eval(),
                          0.0)
      x = tf.constant(0.75)
      self.assertAllClose(logit(x).eval(),
                          1.0986122886681096)
      x = tf.constant(0.99)
      self.assertAllClose(logit(x).eval(),
                          4.5951198501345898)

  def test_logit_1d(self):
    with self.test_session():
      x = tf.constant([0.01, 0.01])
      self.assertAllClose(logit(x).eval(),
                          [-4.5951198501345898, -4.5951198501345898])
      x = tf.constant([0.25, 0.25])
      self.assertAllClose(logit(x).eval(),
                          [-1.0986122886681096, -1.0986122886681096])
      x = tf.constant([0.5, 0.5])
      self.assertAllEqual(logit(x).eval(),
                          [0.0, 0.0])
      x = tf.constant([0.75, 0.75])
      self.assertAllClose(logit(x).eval(),
                          [1.0986122886681096, 1.0986122886681096])
      x = tf.constant([0.99, 0.99])
      self.assertAllClose(logit(x).eval(),
                          [4.5951198501345898, 4.5951198501345898])

  def test_contraint_raises(self):
    with self.test_session():
      x = tf.constant([0.01, -20])
      with self.assertRaisesOpError('Condition'):
        logit(x).eval()
      x = tf.constant([0.01, 20])
      with self.assertRaisesOpError('Condition'):
        logit(x).eval()
      x = tf.constant([0.01, np.inf])
      with self.assertRaisesOpError('Condition'):
        logit(x).eval()
      x = tf.constant([0.01, np.nan])
      with self.assertRaisesOpError('Condition'):
        logit(x).eval()

if __name__ == '__main__':
  tf.test.main()
