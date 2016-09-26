from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import dot


class test_dot_class(tf.test.TestCase):

  def test_dot(self):
    with self.test_session():
      a = tf.constant(np.arange(5, dtype=np.float32))
      b = tf.diag(tf.ones([5]))
      self.assertAllEqual(dot(a, b).eval(),
                          np.dot(a.eval(), b.eval()))
      self.assertAllEqual(dot(b, a).eval(),
                          np.dot(b.eval(), a.eval()))

  def test_all_finite_raises(self):
    with self.test_session():
      a = np.inf * tf.ones([5])
      b = tf.diag(tf.ones([5]))
      with self.assertRaisesOpError('Inf'):
        dot(a, b).eval()
      a = tf.ones([5]) * np.arange(5)
      b = np.inf * tf.diag(tf.ones([5]))
      with self.assertRaisesOpError('Inf'):
        dot(a, b).eval()

if __name__ == '__main__':
  tf.test.main()
