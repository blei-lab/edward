from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import PointMass, Normal
from edward.util import get_irrelevant


class test_get_irrelevant_class(tf.test.TestCase):

  def test_chain_structure(self):
    """a -> b -> c -> d -> e"""
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = Normal(a, 1.0)
      c = Normal(b, 1.0)
      d = Normal(c, 1.0)
      e = Normal(d, 1.0)
      self.assertEqual(set(get_irrelevant(e, d)), set([a, b, c]))
      self.assertEqual(set(get_irrelevant(d, c)), set([a, b]))
      self.assertEqual(get_irrelevant(d, []), [])
      self.assertEqual(get_irrelevant(c, d), [e])
      self.assertEqual(get_irrelevant(a, []), [])
      self.assertEqual(set(get_irrelevant(a, b)), set([c, d, e]))
      self.assertEqual(get_irrelevant(b, d), [e])

  def test_divergent_structure(self):
    """e <- d <- a -> b -> c"""
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = Normal(a, 1.0)
      c = Normal(b, 1.0)
      d = Normal(a, 1.0)
      e = Normal(d, 1.0)
      self.assertEqual(set(get_irrelevant(e, d)), set([a, b, c]))
      self.assertEqual(get_irrelevant(a, d), [e])
      self.assertEqual(set(get_irrelevant(a, [b, d])), set([e, c]))
      self.assertEqual(get_irrelevant(d, b), [c])
      self.assertEqual(get_irrelevant(e, c), [])
      self.assertEqual(get_irrelevant(a, e), [])

  def test_convergent_structure(self):
    """a -> b -> c <- d <- e"""
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = Normal(a, 1.0)
      e = Normal(0.0, 1.0)
      d = Normal(e, 1.0)
      c = Normal(b + d, 1.0)
      self.assertEqual(get_irrelevant(c, []), [])
      self.assertEqual(get_irrelevant(c, b), [a])
      self.assertEqual(set(get_irrelevant(c, [b, d])), set([a, e]))
      self.assertEqual(get_irrelevant(c, e), [])
      self.assertEqual(get_irrelevant(c, [a, e]), [])
      self.assertEqual(set(get_irrelevant(a, b)), set([c, d, e]))
      self.assertEqual(set(get_irrelevant(b, a)), set([d, e]))
      self.assertEqual(get_irrelevant(b, c), [])
      self.assertEqual(set(get_irrelevant(b, a)), set([d, e]))

  def test_functionally_determined_nodes(self):
    """a -> |b| -> c <- d <- e
    Where |x| denotes a node that is functionally determined by its parents.
    """
    with self.test_session():
      a = Normal(0.0, 1.0)
      b = PointMass(a)
      e = Normal(0.0, 1.0)
      d = Normal(e, 1.0)
      c = Normal(b + d, 1.0)
      self.assertEqual(get_irrelevant(c, a), [b])
      self.assertEqual(get_irrelevant(c, e), [])
      self.assertEqual(set(get_irrelevant(b, a)), set([c, d, e]))
      self.assertEqual(get_irrelevant(c, b), [a])
      self.assertEqual(set(get_irrelevant(c, [b, d])), set([a, e]))
      self.assertEqual(set(get_irrelevant(b, [])), set([d, e]))

if __name__ == '__main__':
  tf.test.main()
