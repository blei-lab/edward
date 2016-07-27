from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from edward.util import cumprod

class test_cumprod(tf.test.TestCase):

    def test_cumprod_1d(self):
        with self.test_session():
            x = tf.constant([-1.0, -2.0, -3.0, -4.0])
            self.assertAllEqual(cumprod(x).eval(), 
                                np.array([ -1.,   2.,  -6.,  24.]))
            
    def test_cumprod_2d(self):
        with self.test_session():
            x = tf.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
            self.assertAllClose(cumprod(x).eval(), 
                                np.array([ [-1.],   [2.],  [-6.],  [24.]]))             

    def test_all_finite_raises(self):
        with self.test_session():
            x = np.inf * tf.constant([-1.0, -2.0, -3.0, -4.0])
            with self.assertRaisesOpError('Inf'):
                cumprod(x).eval()
            x = tf.constant([-1.0, np.nan, -3.0, -4.0])
            with self.assertRaisesOpError('NaN'):
                cumprod(x).eval()                

if __name__ == '__main__':
    tf.test.main()
