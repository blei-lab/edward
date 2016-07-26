from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from edward.util import dot

class test_dot(tf.test.TestCase):

  def test_dot(self):
      with self.test_session():
          a = tf.ones([5]) * np.arange(5)
          b = tf.diag(tf.ones([5]))
          self.assertAllEqual(dot(a, b).eval(), 
                              a.eval()[np.newaxis].dot(b.eval()))
          self.assertAllEqual(dot(b, a).eval(), 
                              b.eval().dot(a.eval()[:, np.newaxis]))

  def test_all_finite_raises(self):
      with self.test_session():
          a = np.inf * tf.ones([5]) * np.arange(5)
          b = tf.diag(tf.ones([5])) 
          with self.assertRaisesOpError('Inf'):
              dot(a, b).eval()

if __name__ == '__main__':
  tf.test.main()
