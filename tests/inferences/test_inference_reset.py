"""Test that reset op works."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal


class test_inference_reset_class(tf.test.TestCase):

  def test(self):
    with self.test_session() as sess:
      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=5)

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      inference = ed.KLqp({mu: qmu}, data={x: tf.zeros(5)})
      inference.initialize()
      tf.global_variables_initializer().run()

      first = sess.run(inference.t)
      inference.update()
      second = sess.run(inference.t)
      self.assertEqual(first, second - 1)
      sess.run(inference.reset)
      third = sess.run(inference.t)
      self.assertEqual(first, third)

if __name__ == '__main__':
  tf.test.main()
