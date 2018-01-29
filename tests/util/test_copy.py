from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Categorical, Mixture, Normal


class test_copy_class(tf.test.TestCase):

  def test_scope(self):
    with self.test_session():
      x = tf.constant(2.0)
      x_new = ed.copy(x, scope='new_scope')
      self.assertTrue(x_new.name.startswith('new_scope'))

  def test_replace_itself(self):
    with self.test_session():
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      x_new = ed.copy(x, {x: y}, replace_itself=False)
      self.assertEqual(x_new.eval(), 2.0)
      x_new = ed.copy(x, {x: y}, replace_itself=True)
      self.assertEqual(x_new.eval(), 3.0)

  def test_copy_q(self):
    with self.test_session() as sess:
      x = tf.constant(2.0)
      y = tf.random_normal([])
      x_new = ed.copy(x, {x: y}, replace_itself=True, copy_q=False)
      x_new_val, y_val = sess.run([x_new, y])
      self.assertEqual(x_new_val, y_val)
      x_new = ed.copy(x, {x: y}, replace_itself=True, copy_q=True)
      x_new_val, x_val, y_val = sess.run([x_new, x, y])
      self.assertNotEqual(x_new_val, x_val)
      self.assertNotEqual(x_new_val, y_val)

  def test_copy_parent_rvs(self):
    with self.test_session() as sess:
      x = Normal(0.0, 1.0)
      y = tf.constant(3.0)
      z = x * y
      z_new = ed.copy(z, scope='no_copy_parent_rvs', copy_parent_rvs=False)
      self.assertEqual(len(ed.random_variables()), 1)
      z_new = ed.copy(z, scope='copy_parent_rvs', copy_parent_rvs=True)
      self.assertEqual(len(ed.random_variables()), 2)

  def test_placeholder(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32, name="CustomName")
      y = tf.constant(3.0)
      z = x * y
      z_new = ed.copy(z)
      self.assertEqual(sess.run(z_new, feed_dict={x: 4.0}), 12.0)

  def test_variable(self):
    with self.test_session() as sess:
      x = tf.Variable(2.0, name="CustomName")
      y = tf.constant(3.0)
      z = x * y
      z_new = ed.copy(z)
      tf.variables_initializer([x]).run()
      self.assertEqual(z_new.eval(), 6.0)

  def test_queue(self):
    with self.test_session() as sess:
      tensor = tf.constant([0.0, 1.0, 2.0, 3.0])
      x = tf.train.batch([tensor], batch_size=2, enqueue_many=True,
                         name='CustomName')
      y = tf.constant(3.0)
      z = x * y
      z_new = ed.copy(z)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      self.assertAllEqual(sess.run(z_new), np.array([0.0, 3.0]))
      self.assertAllEqual(sess.run(z_new), np.array([6.0, 9.0]))
      coord.request_stop()
      coord.join(threads)

  def test_list(self):
    with self.test_session() as sess:
      x = Normal(tf.constant(0.0), tf.constant(0.1))
      y = Normal(tf.constant(10.0), tf.constant(0.1))
      cat = Categorical(logits=tf.zeros(5))
      components = [Normal(x, tf.constant(0.1))
                    for _ in range(5)]
      z = Mixture(cat=cat, components=components)
      z_new = ed.copy(z, {x: y.value()})
      self.assertGreater(z_new.value().eval(), 5.0)

  def test_random(self):
    with self.test_session() as sess:
      ed.set_seed(3742)
      x = tf.random_normal([])
      x_copy = ed.copy(x)

      result_copy, result = sess.run([x_copy, x])
      self.assertNotAlmostEquals(result_copy, result)

  def test_scan(self):
    with self.test_session() as sess:
      ed.set_seed(42)
      op = tf.scan(lambda a, x: a + x, tf.constant([2.0, 3.0, 1.0]))
      copy_op = ed.copy(op)

      result_copy, result = sess.run([copy_op, op])
      self.assertAllClose(result_copy, [2.0, 5.0, 6.0])
      self.assertAllClose(result, [2.0, 5.0, 6.0])

  def test_scan_gradients(self):
    with self.test_session() as sess:
      a = tf.Variable([1.0, 2.0, 3.0])
      op = tf.scan(lambda a, x: a + x, a)
      copy_op = ed.copy(op)
      gradient = tf.gradients(op, [a])[0]
      copy_gradient = tf.gradients(copy_op, [a])[0]

      tf.variables_initializer([a]).run()
      result_copy, result = sess.run([copy_gradient, gradient])
      self.assertAllClose(result, [3.0, 2.0, 1.0])
      self.assertAllClose(result_copy, [3.0, 2.0, 1.0])

  def test_nested_scan_gradients(self):
    with self.test_session() as sess:
      a = tf.Variable([1.0, 2.0, 3.0])
      i = tf.constant(0.0)
      tot = tf.constant([0.0, 0.0, 0.0])
      op = tf.while_loop(lambda i, tot: i < 5,
                         lambda i, tot: (i + 1,
                                         tot + tf.scan(lambda x0, x:
                                                       x0 + i * x, a, 0.0)),
                         [i, tot])[1]
      copy_op = ed.copy(op)
      gradient = tf.gradients(op, [a])[0]
      copy_gradient = tf.gradients(copy_op, [a])[0]

      tf.variables_initializer([a]).run()
      result_copy, result = sess.run([copy_gradient, gradient])
      self.assertAllClose(result, [30.0, 20.0, 10.0])
      self.assertAllClose(result_copy, [30.0, 20.0, 10.0])

  def test_swap_tensor_tensor(self):
    with self.test_session():
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = x * y
      qx = tf.constant(4.0)
      z_new = ed.copy(z, {x: qx})
      self.assertEqual(z_new.eval(), 12.0)

  def test_swap_placeholder_tensor(self):
    with self.test_session():
      x = tf.placeholder(tf.float32, name="CustomName")
      y = tf.constant(3.0)
      z = x * y
      qx = tf.constant(4.0)
      z_new = ed.copy(z, {x: qx})
      self.assertEqual(z_new.eval(), 12.0)

  def test_swap_tensor_placeholder(self):
    with self.test_session() as sess:
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = x * y
      qx = tf.placeholder(tf.float32, name="CustomName")
      z_new = ed.copy(z, {x: qx})
      self.assertEqual(sess.run(z_new, feed_dict={qx: 4.0}), 12.0)

  def test_swap_variable_tensor(self):
    with self.test_session():
      x = tf.Variable(2.0, name="CustomName")
      y = tf.constant(3.0)
      z = x * y
      qx = tf.constant(4.0)
      z_new = ed.copy(z, {x: qx})
      tf.variables_initializer([x]).run()
      self.assertEqual(z_new.eval(), 12.0)

  def test_swap_tensor_variable(self):
    with self.test_session() as sess:
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = x * y
      qx = tf.Variable(4.0, name="CustomName")
      z_new = ed.copy(z, {x: qx})
      tf.variables_initializer([qx]).run()
      self.assertEqual(z_new.eval(), 12.0)

  def test_swap_rv_rv(self):
    with self.test_session():
      ed.set_seed(325135)
      x = Normal(0.0, 0.1)
      y = tf.constant(1.0)
      z = x * y
      qx = Normal(10.0, 0.1)
      z_new = ed.copy(z, {x: qx})
      self.assertGreater(z_new.eval(), 5.0)

  def test_swap_rv_tensor(self):
    with self.test_session():
      ed.set_seed(289362)
      x = Normal(0.0, 0.1)
      y = tf.constant(1.0)
      z = x * y
      qx = Normal(10.0, 0.1)
      z_new = ed.copy(z, {x: qx.value()})
      self.assertGreater(z_new.eval(), 5.0)

  def test_swap_tensor_rv(self):
    with self.test_session():
      ed.set_seed(95258)
      x = Normal(0.0, 0.1)
      y = tf.constant(1.0)
      z = x * y
      qx = Normal(10.0, 0.1)
      z_new = ed.copy(z, {x.value(): qx})
      self.assertGreater(z_new.eval(), 5.0)

  def test_ordering_rv_tensor(self):
    # Check that random variables are copied correctly in dependency
    # structure.
    with self.test_session() as sess:
      ed.set_seed(12432)
      x = Bernoulli(logits=0.0)
      y = tf.cast(x, tf.float32)
      y_new = ed.copy(y)
      x_new = ed.copy(x)
      x_new_val, y_new_val = sess.run([x_new, y_new])
      self.assertEqual(x_new_val, y_new_val)

  def test_ordering_rv_rv(self):
    # Check that random variables are copied correctly in dependency
    # structure.
    with self.test_session() as sess:
      ed.set_seed(21782)
      x = Normal(loc=0.0, scale=10.0)
      x_abs = tf.abs(x)
      y = Normal(loc=x_abs, scale=1e-8)
      y_new = ed.copy(y)
      x_new = ed.copy(x)
      x_new_val, y_new_val = sess.run([x_new, y_new])
      self.assertAllClose(abs(x_new_val), y_new_val)

if __name__ == '__main__':
  tf.test.main()
