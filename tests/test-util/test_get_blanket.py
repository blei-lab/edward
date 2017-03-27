from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import get_blanket


class test_get_blanket_class(tf.test.TestCase):

  def test_blanket_structure(self):
    """a -> c <- b
            |
            v
       d -> f <- e
    """
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = Normal(0.0, 1.0)
      c = Normal(a * b, 1.0)
      d = Normal(0.0, 1.0)
      e = Normal(0.0, 1.0)
      f = Normal(c * d * e, 1.0)
      self.assertEqual(set(get_blanket(a)), set([b, c]))
      self.assertEqual(set(get_blanket(b)), set([a, c]))
      self.assertEqual(set(get_blanket(c)), set([a, b, d, e, f]))
      self.assertEqual(set(get_blanket(d)), set([c, e, f]))
      self.assertEqual(set(get_blanket(e)), set([c, d, f]))
      self.assertEqual(set(get_blanket(f)), set([c, d, e]))

if __name__ == '__main__':
  tf.test.main()
