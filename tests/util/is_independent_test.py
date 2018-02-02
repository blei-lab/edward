from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import Normal
from edward.util import is_independent


class test_is_independent_class(tf.test.TestCase):

  def test_chain_structure(self):
    """a -> b -> c -> d -> e"""
    a = Normal(0.0, 1.0)
    b = Normal(a, 1.0)
    c = Normal(b, 1.0)
    d = Normal(c, 1.0)
    e = Normal(d, 1.0)
    self.assertTrue(is_independent(c, e, d))
    self.assertTrue(is_independent([a, b, c], e, d))
    self.assertTrue(is_independent([a, b], [d, e], c))
    self.assertFalse(is_independent([a, b, e], d, c))

  def test_binary_structure(self):
    """f <- c <- a -> b -> d
            |         |
            v         v
            g         e
    """
    a = Normal(0.0, 1.0)
    b = Normal(a, 1.0)
    c = Normal(a, 1.0)
    d = Normal(b, 1.0)
    e = Normal(b, 1.0)
    f = Normal(c, 1.0)
    g = Normal(c, 1.0)
    self.assertFalse(is_independent(b, c))
    self.assertTrue(is_independent(b, c, a))
    self.assertTrue(is_independent(d, [a, c, e, f, g], b))
    self.assertFalse(is_independent(b, [e, d], a))
    self.assertFalse(is_independent(a, [b, c, d, e, f, g]))

  def test_grid_structure(self):
    """a -> b -> c
       |    |    |
       v    v    v
       d -> e -> f
    """
    a = Normal(0.0, 1.0)
    b = Normal(a, 1.0)
    c = Normal(b, 1.0)
    d = Normal(a, 1.0)
    e = Normal(b + d, 1.0)
    f = Normal(e + c, 1.0)
    self.assertFalse(is_independent(f, [a, b, d]))
    self.assertTrue(is_independent(f, [a, b, d], [e, c]))
    self.assertTrue(is_independent(e, [a, c], [b, d]))
    self.assertFalse(is_independent(e, f, [b, d]))
    self.assertFalse(is_independent(e, f, [a, b, c, d]))

if __name__ == '__main__':
  tf.test.main()
