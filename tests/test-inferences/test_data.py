from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import Normal
from edward.stats import norm


class NormalNormal:
  """p(x, mu) = Normal(x | mu, 1) Normal(mu | 1, 1)"""
  def log_prob(self, xs, zs):
    log_prior = norm.logpdf(zs['mu'], 1.0, 1.0)
    log_lik = tf.reduce_sum(norm.logpdf(xs['x'], zs['mu'], 1.0))
    return log_lik + log_prior


class test_inference_data_class(tf.test.TestCase):

  def read_and_decode_single_example(self, filename):
    # Construct a queue containing a list of filenames.
    filename_queue = tf.train.string_input_producer([filename])
    # Read a single serialized example from a filename.
    # `serialized_example` is a Tensor of type str.
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Convert serialized example back to actual values,
    # describing format of the objects to be returned.
    features = tf.parse_single_example(
        serialized_example,
        features={'outcome': tf.FixedLenFeature([], tf.int64)})
    return features['outcome']

  def _test(self, sess, x_data, n_minibatch, x_val=None, is_file=False):
    model = NormalNormal()

    qmu = Normal(mu=tf.Variable(0.0), sigma=tf.constant(1.0))

    data = {'x': x_data}
    inference = ed.KLqp({'mu': qmu}, data, model_wrapper=model)
    inference.initialize(n_minibatch=n_minibatch)

    init = tf.global_variables_initializer()
    init.run()

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if x_val is not None:
      # Placeholder setting.
      # Check data is same as data fed to it.
      feed_dict = {inference.data['x']: x_val}
      # avoid directly fetching placeholder
      data_id = [tf.identity(v) for v in six.itervalues(inference.data)]
      val = sess.run(data_id, feed_dict)
      assert np.all(val == x_val)
    elif is_file:
      # File reader setting.
      # Check data varies by session run.
      val = sess.run(inference.data['x'])
      val_1 = sess.run(inference.data['x'])
      assert not np.all(val == val_1)
    elif n_minibatch is None:
      # Preloaded full setting.
      # Check data is full data.
      val = sess.run(inference.data['x'])
      assert np.all(val == data['x'])
    elif n_minibatch == 1:
      # Preloaded batch setting, with n_minibatch=1.
      # Check data is randomly shuffled.
      assert not np.all([sess.run(inference.data)['x'] == data['x'][i]
                         for i in range(10)])
    else:
      # Preloaded batch setting.
      # Check data is randomly shuffled.
      val = sess.run(inference.data)
      assert not np.all(val['x'] == data['x'][:n_minibatch])
      # Check data varies by session run.
      val_1 = sess.run(inference.data)
      assert not np.all(val['x'] == val_1['x'])

    inference.finalize()

    coord.request_stop()
    coord.join(threads)

  def test_preloaded_full(self):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      self._test(sess, x_data, None)

  def test_preloaded_batch_1(self):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      self._test(sess, x_data, 1)

  def test_preloaded_batch_5(self):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      self._test(sess, x_data, 5)

  def test_feeding(self):
    with self.test_session() as sess:
      x_val = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      x_data = tf.placeholder(tf.float32)
      self._test(sess, x_data, None, x_val)

  def test_read_file(self):
    with self.test_session() as sess:
      x_data = self.read_and_decode_single_example(
          "tests/data/toy_data.tfrecords")
      self._test(sess, x_data, None, is_file=True)

if __name__ == '__main__':
  ed.set_seed(1512351)
  tf.test.main()
