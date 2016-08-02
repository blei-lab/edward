from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import edward as ed
import six

from edward.models import Variational, Normal
from edward.stats import norm

ed.set_seed(1512351)

class NormalModel:
    """
    p(x, z) = Normal(x; z, 1) Normal(z; 0, 1)
    """
    def log_prob(self, xs, zs):
        log_prior = norm.logpdf(zs, 0.0, 1.0)
        log_lik = tf.pack([tf.reduce_sum(norm.logpdf(xs['x'], z, 1.0))
                           for z in tf.unpack(zs)])
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
        features = tf.parse_single_example(serialized_example,
            features={'outcome': tf.FixedLenFeature([], tf.int64)})
        return features['outcome']


    def _test(self, sess, data, n_minibatch, x=None, is_file=False):
        model = NormalModel()
        variational = Variational()
        variational.add(Normal())

        inference = ed.MFVI(model, variational, data)
        inference.initialize(n_minibatch=n_minibatch)

        if x is not None:
            # Placeholder setting.
            # Check data is same as data fed to it.
            feed_dict = {inference.data['x']: x}
            # avoid directly fetching placeholder
            data_id = {k: tf.identity(v) for k,v in
                       six.iteritems(inference.data)}
            val = sess.run(data_id, feed_dict)
            assert np.all(val['x'] == x)
        elif is_file:
            # File reader setting.
            # Check data varies by session run.
            val = sess.run(inference.data)
            val_1 = sess.run(inference.data)
            assert not np.all(val['x'] == val_1['x'])
        elif n_minibatch is None:
            # Preloaded full setting.
            # Check data is full data.
            val = sess.run(inference.data)
            assert np.all(val['x'] == data['x'])
        else:
            # Preloaded batch setting.
            # Check data is randomly shuffled.
            val = sess.run(inference.data)
            assert not np.all(val['x'] == data['x'][:n_minibatch])
            # Check data varies by session run.
            val_1 = sess.run(inference.data)
            assert not np.all(val['x'] == val_1['x'])

        inference.finalize()

    def test_preloaded_full(self):
        with self.test_session() as sess:
            data = {'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
            self._test(sess, data, None)

    def test_preloaded_batch_1(self):
        with self.test_session() as sess:
            data = {'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
            self._test(sess, data, 1)
        
    def test_preloaded_batch_5(self):
        with self.test_session() as sess:
            data = {'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
            self._test(sess, data, 5)
    
    def test_feeding(self):
        with self.test_session() as sess:
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            data = {'x': tf.placeholder(tf.float32)}
            self._test(sess, data, None, x)

    def test_read_file(self):
        with self.test_session() as sess:
            x = self.read_and_decode_single_example("tests/data/toy_data.tfrecords")
            data = {'x': x}
            self._test(sess, data, None, is_file=True)

if __name__ == '__main__':
    tf.test.main()
