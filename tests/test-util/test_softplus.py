from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from edward.util import softplus

class test_softplus_class(tf.test.TestCase):

    def test_softplus_0d(self):
        with self.test_session():
            x = tf.constant(-1e32)
            self.assertAllEqual(softplus(x).eval(), 
                             0.0)   
            x = tf.constant(0.0)
            self.assertAllClose(softplus(x).eval(), 
                              0.69314718055994529)                                       
            x = tf.constant(1.0)
            self.assertAllClose(softplus(x).eval(), 
                             1.3132616875182228)
            x = tf.constant(10.0)
            self.assertAllClose(softplus(x).eval(), 
                             10.000045398899218)
            x = tf.constant(100.0)
            self.assertAllEqual(softplus(x).eval(), 
                             100.0)            

    def test_softplus_1d(self):
        with self.test_session():
            x = tf.constant([-1e32, -1e32])
            self.assertAllEqual(softplus(x).eval(), 
                             [0.0, 0.0])   
            x = tf.constant([0.0, 0.0])
            self.assertAllClose(softplus(x).eval(), 
                              [0.69314718055994529, 0.69314718055994529])                                       
            x = tf.constant([1.0, 1.0])
            self.assertAllClose(softplus(x).eval(), 
                             [1.3132616875182228, 1.3132616875182228])
            x = tf.constant([10.0, 10.0])
            self.assertAllClose(softplus(x).eval(), 
                             [10.000045398899218, 10.000045398899218])
            x = tf.constant([100.0, 100.0])
            self.assertAllEqual(softplus(x).eval(), 
                             [100.0, 100.0])   
            
    def test_contraint_raises(self):
        with self.test_session():
            x = tf.constant([0.01, np.inf])
            with self.assertRaisesOpError('Inf'):
                softplus(x).eval()
            x = tf.constant([0.01, np.nan])
            with self.assertRaisesOpError('NaN'):
                softplus(x).eval()             

if __name__ == '__main__':
    tf.test.main()
