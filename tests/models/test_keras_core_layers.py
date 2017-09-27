from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import keras.layers as layers
import tensorflow as tf

from edward.models import Normal


class test_keras_core_layers_class(tf.test.TestCase):

  def test_dense(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Dense(32)(x)

  def test_activation(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Activation('tanh')(x)

  def test_dropout(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Dropout(0.5)(x)

  def test_flatten(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Flatten()(x)
    with self.test_session():
      self.assertEqual(y.eval().shape, (100, 50))

  def test_reshape(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Reshape((5, 10))(x)
    with self.test_session():
      self.assertEqual(y.eval().shape, (100, 5, 10))

  def test_permute(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Permute((2, 1))(x)
    with self.test_session():
      self.assertEqual(y.eval().shape, (100, 5, 10))

  def test_repeat_vector(self):
    x = Normal(loc=tf.zeros([100, 10]), scale=tf.ones([100, 10]))
    y = layers.RepeatVector(2)(x)
    with self.test_session():
      self.assertEqual(y.eval().shape, (100, 2, 10))

  def test_lambda(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Lambda(lambda x: x ** 2)(x)

  def test_activity_regularization(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.ActivityRegularization(l1=0.1)(x)

  def test_masking(self):
    x = Normal(loc=tf.zeros([100, 10, 5]), scale=tf.ones([100, 10, 5]))
    y = layers.Masking()(x)

if __name__ == '__main__':
  tf.test.main()
