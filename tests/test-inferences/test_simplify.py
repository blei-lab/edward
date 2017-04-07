from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import edward as ed
import numpy as np
import tensorflow as tf

from edward.inferences.conjugacy import simplify as simplify


class test_simplify_class(tf.test.TestCase):

  def test_mul_n(self):
    a = tf.constant(1.)
    b = tf.constant(2.)
    c = tf.constant(3.)
    ab = simplify._mul_n([a, b])
    abc = simplify._mul_n([a, b, c])

    sess = tf.InteractiveSession()
    self.assertEqual(sess.run(ab), 2.)
    self.assertEqual(sess.run(abc), 6.)

  def test_as_float(self):
    self.assertEqual(simplify.as_float(1), 1.)
    self.assertIsNone(simplify.as_float('one'))

  def test_identity_op_simplify(self):
    expr = ('#Identity', ('#Mul', ('#Identity', ('#x',)),
                          ('#Identity', (3.7,))))
    did_something, new_expr = simplify.identity_op_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Mul', ('#x',), (3.7,)))
    did_something, new_expr = simplify.power_op_simplify(new_expr)
    self.assertFalse(did_something)

  def test_pow_simplify_and_power_op_simplify(self):
    expr = ('#Square', ('#Reciprocal', ('#Sqrt', ('#x',))))
    did_something, new_expr = simplify.power_op_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr,
                     ('#CPow2.0000e+00',
                      ('#CPow-1.0000e+00', ('#CPow5.0000e-01', ('#x',)))))
    did_something, new_expr = simplify.power_op_simplify(new_expr)
    self.assertFalse(did_something)

    did_something, new_expr = simplify.pow_simplify(new_expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#CPow-1.0000e+00', ('#x',)))
    did_something, new_expr = simplify.pow_simplify(new_expr)
    self.assertFalse(did_something)

  def test_log_cpow_simplify(self):
    expr = ('#Log', ('#CPow2.3000e+01', ('#x',)))
    did_something, new_expr = simplify.log_pow_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Mul', (2.3000e+01,), ('#Log', ('#x',))))
    did_something, new_expr = simplify.log_pow_simplify(new_expr)
    self.assertFalse(did_something)

  def test_log_pow_simplify(self):
    expr = ('#Mul', (3.3,), ('#Log', ('#Pow', (2.3,), (1.3,))))
    did_something, new_expr = simplify.log_pow_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Mul', (3.3,), ('#Mul', (1.3,),
                                                 ('#Log', (2.3,)))))
    did_something, new_expr = simplify.log_pow_simplify(new_expr)
    self.assertFalse(did_something)

  def test_log_mul_simplify(self):
    expr = ('#Log', ('#Mul', (3,), (4.2,), (1.2e+01,), ('#x',)))
    did_something, new_expr = simplify.log_mul_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Add', ('#Log', (3,)),
                                ('#Log', (4.2,)), ('#Log', (1.2e+01,)),
                                ('#Log', ('#x',))))
    did_something, new_expr = simplify.log_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_cpow_mul_simplify(self):
    expr = ('#CPow2.1', ('#Mul', (3,), (4.,), (1.2e+01,)))
    did_something, new_expr = simplify.pow_mul_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Mul', ('#CPow2.1', (3,)),
                                ('#CPow2.1', (4.,)),
                                ('#CPow2.1', (1.2e+01,))))
    did_something, new_expr = simplify.pow_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_pow_mul_simplify(self):
    expr = ('#Pow', ('#Mul', (3,), (4.,), (1.2e+01,)), (2.1,))
    did_something, new_expr = simplify.pow_mul_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Mul', ('#Pow', (3,), (2.1,)),
                                ('#Pow', (4.,), (2.1,)),
                                ('#Pow', (1.2e+01,), (2.1,))))
    did_something, new_expr = simplify.pow_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_add_simplify(self):
    expr = ('#Mul', ('#Add', (3.,), (2.,)),
            ('#Add', (4.,), (5.,)))
    did_something, new_expr = simplify.mul_add_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Add', ('#Add', ('#Mul', (3.,), (4.,)),
                                         ('#Mul', (3.,), (5.,))),
                                ('#Add', ('#Mul', (2.,), (4.,)),
                                 ('#Mul', (2.,), (5.,)))))
    did_something, new_expr = simplify.pow_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_add_add_simplify(self):
    expr = ('#Add', (3.,), ('#Add', (4.,), (5.,), ('#Add', (6.,))))
    did_something, new_expr = simplify.add_add_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Add', (3.,), (4.,), (5.,), (6.,)))
    did_something, new_expr = simplify.add_add_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_mul_simplify(self):
    expr = ('#Mul', (3.,), ('#Mul', (4.,), (5.,), ('#Mul', (6.,))))
    did_something, new_expr = simplify.mul_mul_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Mul', (3.,), (4.,), (5.,), (6.,)))
    did_something, new_expr = simplify.mul_mul_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_one_simplify(self):
    expr = ('#Mul', (3.,), (1.,), (4.,), (5.,), (6.,), (1.,))
    did_something, new_expr = simplify.mul_one_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Mul', (3.,), (4.,), (5.,), (6.,)))
    did_something, new_expr = simplify.mul_one_simplify(new_expr)
    self.assertFalse(did_something)

  def test_add_zero_simplify(self):
    expr = ('#Add', (3.,), (0.,), (4.,), (5.,), (6.,), (0.,))
    did_something, new_expr = simplify.add_zero_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Add', (3.,), (4.,), (5.,), (6.,)))
    did_something, new_expr = simplify.add_zero_simplify(new_expr)
    self.assertFalse(did_something)

  def test_mul_zero_simplify(self):
    expr = ('#Mul', (3.,), (0.,), (5.,), (6.,), (1.,))
    did_something, new_expr = simplify.mul_zero_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, (0,))
    did_something, new_expr = simplify.mul_zero_simplify(new_expr)
    self.assertFalse(did_something)

  def test_square_add_simplify(self):
    expr = ('#CPow2.0000e+00', ('#Add', (1.,), ('#Neg', (2.,))))
    did_something, new_expr = simplify.square_add_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Add', ('#CPow2.0000e+00', (1.,)),
                                ('#Mul', (2.0,), (1.,), ('#Neg', (2.,))),
                                ('#CPow2.0000e+00', ('#Neg', (2.,)))))
    did_something, new_expr = simplify.square_add_simplify(new_expr)
    self.assertFalse(did_something)

  def test_expr_contains(self):
    expr = ('#Add', ('#Mul', (1.5, ('#CPow1.2000e+00', ('#x',)))),
            ('#Mul', (1.2,), (7,)))
    self.assertTrue(simplify.expr_contains(expr, '#x'))
    self.assertFalse(simplify.expr_contains(expr, '#CPow'))
    self.assertTrue(simplify.expr_contains(expr, '#CPow1.2000e+00'))
    self.assertTrue(simplify.expr_contains(expr, '#Mul'))
    self.assertTrue(simplify.expr_contains(expr, '#Add'))
    self.assertTrue(simplify.expr_contains(expr[1], '#x'))
    self.assertFalse(simplify.expr_contains(expr[2], '#x'))

  def test_add_const_simplify(self):
    expr = ('#Add', ('#Mul', (1.5, ('#CPow1.2000e+00', ('#x',)))),
            ('#Mul', (1.2,), (7,)))
    did_something, new_expr = simplify.add_const_simplify(expr)
    self.assertTrue(did_something)
    self.assertEqual(new_expr, ('#Add', ('#Mul',
                                         (1.5,
                                          ('#CPow1.2000e+00', ('#x',))))))
    did_something, new_expr = simplify.add_const_simplify(new_expr)
    self.assertFalse(did_something)


if __name__ == '__main__':
  tf.test.main()
