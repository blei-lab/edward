from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, PointMass


class test_saver_class(tf.test.TestCase):

  def test_export_meta_graph(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      qmu = PointMass(params=tf.Variable(1.0))

      inference = ed.MAP({mu: qmu}, data={x: x_data})
      inference.run(n_iter=10)

      saver = tf.train.Saver()
      saver.export_meta_graph("/tmp/test_saver.meta")

  def test_import_meta_graph(self):
    with self.test_session() as sess:
      new_saver = tf.train.import_meta_graph("tests/data/test_saver.meta")
      new_saver.restore(sess, "tests/data/test_saver")
      qmu_variable = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope="posterior")[0]
      self.assertNotEqual(qmu_variable.eval(), 1.0)

  def test_restore(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      with tf.variable_scope("posterior"):
        qmu = PointMass(params=tf.Variable(1.0))

      inference = ed.MAP({mu: qmu}, data={x: x_data})

      saver = tf.train.Saver()
      saver.restore(sess, "tests/data/test_saver")
      qmu_variable = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope="posterior")[0]
      self.assertNotEqual(qmu_variable.eval(), 1.0)

  def test_save(self):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      with tf.variable_scope("posterior"):
        qmu = PointMass(params=tf.Variable(1.0))

      inference = ed.MAP({mu: qmu}, data={x: x_data})
      inference.run(n_iter=10)

      saver = tf.train.Saver()
      saver.save(sess, "/tmp/test_saver")

if __name__ == '__main__':
  ed.set_seed(23451)
  tf.test.main()
