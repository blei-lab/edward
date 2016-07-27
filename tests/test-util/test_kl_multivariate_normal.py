from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from edward.util import kl_multivariate_normal

class test_kl_multivariate_normal(tf.test.TestCase):

    def test_kl_multivariate_normal_0d(self):
        with self.test_session():
            loc_one   = tf.constant(0.0)
            scale_one = tf.constant(1.0)
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                0.0)
            loc_one   = tf.constant(10.0)
            scale_one = tf.constant(2.0)
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                50.806854)
            loc_one   = tf.constant(0.0)
            scale_one = tf.constant(1.0)
            loc_two   = tf.constant(0.0)
            scale_two = tf.constant(1.0)
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one,
                                                       loc_two=loc_two,
                                                       scale_two=scale_two).eval(), 
                                0.0)
            loc_one   = tf.constant(10.0)
            scale_one = tf.constant(2.0)
            loc_two   = tf.constant(10.0)
            scale_two = tf.constant(5.0)
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one,
                                                       loc_two=loc_two,
                                                       scale_two=scale_two).eval(), 
                                0.496290802)

    def test_kl_multivariate_normal_1d(self):
        with self.test_session():
            loc_one   = tf.constant([0.0])
            scale_one = tf.constant([1.0])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                0.0)
            loc_one   = tf.constant([10.0])
            scale_one = tf.constant([2.0])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                50.806854)
            loc_one   = tf.constant([10.0])
            scale_one = tf.constant([2.0])
            loc_two   = tf.constant([10.0])
            scale_two = tf.constant([2.0])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one,
                                                       loc_two=loc_two,
                                                       scale_two=scale_two).eval(), 
                                0.0)
            loc_one   = tf.constant([0.0, 0.0])
            scale_one = tf.constant([1.0, 1.0])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                0.0)
            loc_one   = tf.constant([10.0, 10.0])
            scale_one = tf.constant([2.0 , 2.0 ])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                101.61370849)    
            loc_one   = tf.constant([10.0, 10.0])
            scale_one = tf.constant([2.0 , 2.0 ])
            loc_two   = tf.constant([9.0, 9.0])
            scale_two = tf.constant([1.0 , 1.0 ])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one,
                                                       loc_two=loc_two,
                                                       scale_two=scale_two).eval(), 
                                2.6137056350)             

    def test_kl_multivariate_normal_2d(self):
        with self.test_session():
            loc_one   = tf.constant([[0.0, 0.0], [0.0, 0.0]])
            scale_one = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                0.0)
            loc_one   = tf.constant([[10.0, 10.0], [10.0, 10.0]])
            scale_one = tf.constant([[2.0 , 2.0 ], [2.0 , 2.0 ]])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one).eval(), 
                                203.22741699)
            loc_one   = tf.constant([[10.0, 10.0], [10.0, 10.0]])
            scale_one = tf.constant([[2.0 , 2.0 ], [2.0 , 2.0 ]])
            loc_two   = tf.constant([[10.0, 10.0], [10.0, 10.0]])
            scale_two = tf.constant([[2.0 , 2.0 ], [2.0 , 2.0 ]])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one,
                                                       loc_two=loc_two,
                                                       scale_two=scale_two).eval(), 
                                0.0)
            loc_one   = tf.constant([[10.0, 10.0], [10.0, 10.0]])
            scale_one = tf.constant([[2.0 , 2.0 ], [2.0 , 2.0 ]])
            loc_two   = tf.constant([[9.0, 9.0], [9.0, 9.0]])
            scale_two = tf.constant([[3.0 , 3.0 ], [3.0 , 3.0 ]])
            self.assertAllClose(kl_multivariate_normal(loc_one,  
                                                       scale_one,
                                                       loc_two=loc_two,
                                                       scale_two=scale_two).eval(), 
                                0.7329716682)

    def test_contraint_raises(self):
        with self.test_session():
            loc_one   = tf.constant(10.0)
            scale_one = tf.constant(-1.0)
            loc_two   = tf.constant(10.0)
            scale_two = tf.constant(-1.0)
            with self.assertRaisesOpError('Condition'):
                kl_multivariate_normal(loc_one,  
                                       scale_one).eval()
                kl_multivariate_normal(loc_one,  
                                       scale_one,
                                       loc_two=loc_two,
                                       scale_two=scale_two).eval()
            loc_one   = np.inf * tf.constant(10.0)
            scale_one = tf.constant(1.0)
            loc_two   = tf.constant(10.0)
            scale_two = tf.constant(1.0)
            with self.assertRaisesOpError('Inf'):
                kl_multivariate_normal(loc_one,  
                                       scale_one).eval()
                kl_multivariate_normal(loc_one,  
                                       scale_one,
                                       loc_two=loc_two,
                                       scale_two=scale_two).eval()
            loc_one   = tf.constant(10.0)
            scale_one = tf.constant(1.0)
            loc_two   = np.nan * tf.constant(10.0)
            scale_two = tf.constant(1.0)
            with self.assertRaisesOpError('NaN'):
                kl_multivariate_normal(loc_one,  
                                       scale_one).eval()
                kl_multivariate_normal(loc_one,  
                                       scale_one,
                                       loc_two=loc_two,
                                       scale_two=scale_two).eval()
                           

if __name__ == '__main__':
    tf.test.main()
