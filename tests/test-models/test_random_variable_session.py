from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal


class test_random_variable_session_class(tf.test.TestCase):

  def test_eval(self):
    with self.test_session() as sess:
      x = Normal(0.0, 0.1)
      x_ph = tf.placeholder(tf.float32, [])
      y = Normal(x_ph, 0.1)
      self.assertLess(x.eval(), 5.0)
      self.assertLess(x.eval(sess), 5.0)
      self.assertLess(x.eval(feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(y.eval(feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(y.eval(sess, feed_dict={x_ph: 100.0}), 5.0)
      self.assertRaises(tf.errors.InvalidArgumentError, y.eval)
      self.assertRaises(tf.errors.InvalidArgumentError, y.eval, sess)

  def test_run(self):
    with self.test_session() as sess:
      x = Normal(0.0, 0.1)
      x_ph = tf.placeholder(tf.float32, [])
      y = Normal(x_ph, 0.1)
      self.assertLess(sess.run(x), 5.0)
      self.assertLess(sess.run(x, feed_dict={x_ph: 100.0}), 5.0)
      self.assertGreater(sess.run(y, feed_dict={x_ph: 100.0}), 5.0)
      self.assertRaises(tf.errors.InvalidArgumentError, sess.run, y)

if __name__ == '__main__':
  ed.set_seed(82341)
  tf.test.main()
