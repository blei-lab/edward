from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Categorical, Mixture, Normal
from edward.util import copy, set_seed


class test_copy_class(tf.test.TestCase):

  def test_placeholder(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32, name="CustomName")
      y = tf.constant(3.0)
      z = x * y
      z_new = copy(z)
      self.assertEqual(sess.run(z_new, feed_dict={x: 4.0}), 12.0)

  def test_variable(self):
    with self.test_session() as sess:
      x = tf.Variable(2.0, name="CustomName")
      y = tf.constant(3.0)
      z = x * y
      z_new = copy(z)
      tf.variables_initializer([x]).run()
      self.assertEqual(z_new.eval(), 6.0)

  def test_queue(self):
    with self.test_session() as sess:
      tensor = tf.constant([0.0, 1.0, 2.0, 3.0])
      x = tf.train.batch([tensor], batch_size=2, enqueue_many=True,
                         name='CustomName')
      y = tf.constant(3.0)
      z = x * y
      z_new = copy(z)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      self.assertAllEqual(sess.run(z_new), np.array([0.0, 3.0]))
      self.assertAllEqual(sess.run(z_new), np.array([6.0, 9.0]))
      coord.request_stop()
      coord.join(threads)

  def test_list(self):
    with self.test_session() as sess:
      x = Normal(mu=tf.constant(0.0), sigma=tf.constant(0.1))
      y = Normal(mu=tf.constant(10.0), sigma=tf.constant(0.1))
      cat = Categorical(logits=tf.zeros(5))
      components = [Normal(mu=x, sigma=tf.constant(0.1))
                    for _ in range(5)]
      z = Mixture(cat=cat, components=components)
      z_new = copy(z, {x: y.value()})
      self.assertGreater(z_new.value().eval(), 5.0)

  def test_tensor_tensor(self):
    with self.test_session():
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = x * y
      qx = tf.constant(4.0)
      z_new = copy(z, {x: qx})
      self.assertEqual(z_new.eval(), 12.0)

  # def test_variable_tensor(self):
  #   with self.test_session():
  #     x = tf.Variable(2.0, name="CustomName")
  #     y = tf.constant(3.0)
  #     z = x * y
  #     qx = tf.constant(4.0)
  #     z_new = copy(z, {x: qx})
  #     self.assertEqual(z_new.eval(), 12.0)

  def test_tensor_variable(self):
    with self.test_session() as sess:
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = x * y
      qx = tf.Variable(4.0, name="CustomName")
      z_new = copy(z, {x: qx})
      tf.variables_initializer([qx]).run()
      self.assertEqual(z_new.eval(), 12.0)

  def test_placeholder_tensor(self):
    with self.test_session():
      x = tf.placeholder(tf.float32, name="CustomName")
      y = tf.constant(3.0)
      z = x * y
      qx = tf.constant(4.0)
      z_new = copy(z, {x: qx})
      self.assertEqual(z_new.eval(), 12.0)

  def test_tensor_placeholder(self):
    with self.test_session() as sess:
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = x * y
      qx = tf.placeholder(tf.float32, name="CustomName")
      z_new = copy(z, {x: qx})
      self.assertEqual(sess.run(z_new, feed_dict={qx: 4.0}), 12.0)

  def test_dict_rv_rv(self):
    with self.test_session():
      set_seed(325135)
      x = Normal(mu=0.0, sigma=0.1)
      y = tf.constant(1.0)
      z = x * y
      qx = Normal(mu=10.0, sigma=0.1)
      z_new = copy(z, {x: qx})
      self.assertGreater(z_new.eval(), 5.0)

  def test_dict_rv_tensor(self):
    with self.test_session():
      set_seed(289362)
      x = Normal(mu=0.0, sigma=0.1)
      y = tf.constant(1.0)
      z = x * y
      qx = Normal(mu=10.0, sigma=0.1)
      z_new = copy(z, {x: qx.value()})
      self.assertGreater(z_new.eval(), 5.0)

  def test_dict_tensor_rv(self):
    with self.test_session():
      set_seed(95258)
      x = Normal(mu=0.0, sigma=0.1)
      y = tf.constant(1.0)
      z = x * y
      qx = Normal(mu=10.0, sigma=0.1)
      z_new = copy(z, {x.value(): qx})
      self.assertGreater(z_new.eval(), 5.0)

  def test_scan(self):
    with self.test_session():
      set_seed(42)
      op = tf.scan(lambda a, x: a + x, tf.constant([2.0, 3.0, 1.0]))

      self.assertAllClose(op.eval(), [2.0, 5.0, 6.0])
      self.assertAllClose(copy(op).eval(), [2.0, 5.0, 6.0])

  def test_scan_random(self):
    with self.test_session() as session:
      set_seed(1234)
      op = tf.scan(lambda a, x: a + x, tf.random_normal([3]))
      copy_op = copy(op)

      result = session.run([copy_op, copy_op, op, op])
      self.assertAllClose(result[0], result[1])
      self.assertAllClose(result[2], result[3])

      # currently set_seed does seem to prevent variate generation to work
      # self.assertNotAlmostEquals(result[0][0], result[2][0])
      # self.assertNotAlmostEquals(result[0][1], result[2][1])
      # self.assertNotAlmostEquals(result[0][2], result[2][2])


if __name__ == '__main__':
  tf.test.main()
