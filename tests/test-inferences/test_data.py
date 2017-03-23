from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import Normal


class test_inference_data_class(tf.test.TestCase):

  def test_preloaded_full(self):
    with self.test_session() as sess:
      x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones(10) * mu, scale=tf.ones(1))

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      inference = ed.KLqp({mu: qmu}, data={x: x_data})
      inference.initialize()
      tf.global_variables_initializer().run()

      val = sess.run(inference.data[x])
      self.assertAllEqual(val, x_data)

  def test_feeding(self):
    with self.test_session() as sess:
      x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
      x_ph = tf.placeholder(tf.float32, [10])

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones(10) * mu, scale=tf.ones(10))

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      inference = ed.KLqp({mu: qmu}, data={x: x_ph})
      inference.initialize()
      tf.global_variables_initializer().run()

      val = sess.run(  # avoid directly fetching placeholder
          tf.identity(list(six.itervalues(inference.data))[0]),
          feed_dict={inference.data[x]: x_data})
      self.assertAllEqual(val, x_data)

  def test_read_file(self):
    with self.test_session() as sess:
      # Construct a queue containing a list of filenames.
      filename_queue = tf.train.string_input_producer(
          ["tests/data/toy_data.tfrecords"])
      # Read a single serialized example from a filename.
      # `serialized_example` is a Tensor of type str.
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      # Convert serialized example back to actual values,
      # describing format of the objects to be returned.
      features = tf.parse_single_example(
          serialized_example,
          features={'outcome': tf.FixedLenFeature([], tf.float32)})
      x_batch = features['outcome']

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones([]) * mu, scale=tf.ones([]))

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      inference = ed.KLqp({mu: qmu}, data={x: x_batch})
      inference.initialize(scale={x: 10.0})

      tf.global_variables_initializer().run()

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      # Check data varies by session run.
      val = sess.run(inference.data[x])
      val_1 = sess.run(inference.data[x])
      self.assertNotEqual(val, val_1)

      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  ed.set_seed(1512351)
  tf.test.main()
