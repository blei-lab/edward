from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import reduce_logmeanexp


class test_reduce_logmeanexp_class(tf.test.TestCase):

  def test_reduce_logmeanexp_1d(self):
    with self.test_session():
      x = tf.constant([-1.0, -2.0, -3.0, -4.0])
      self.assertAllClose(reduce_logmeanexp(x).eval(),
                          -1.9461046625586951)

  def test_reduce_logmeanexp_2d(self):
    with self.test_session():
      x = tf.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
      self.assertAllClose(reduce_logmeanexp(x).eval(),
                          -1.9461046625586951)
      x = tf.constant([[-1.0, -2.0], [-3.0, -4.0]])
      self.assertAllClose(reduce_logmeanexp(x).eval(),
                          -1.9461046625586951)
      self.assertAllClose(reduce_logmeanexp(x, 0).eval(),
                          np.array([-1.5662191695169727,
                                    -2.5662191695169727]))
      self.assertAllClose(reduce_logmeanexp(x, 1).eval(),
                          np.array([-1.3798854930417224,
                                    -3.3798854930417224]))

if __name__ == '__main__':
  tf.test.main()
