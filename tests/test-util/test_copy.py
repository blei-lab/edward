from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Mixture, Normal


class test_copy_class(tf.test.TestCase):

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


if __name__ == '__main__':
  tf.test.main()
