import tensorflow as tf

from blackbox.util import get_dims

x = tf.constant(0.0)
print get_dims(x)

x = tf.zeros([2])
print get_dims(x)

x = tf.zeros([2, 2])
print get_dims(x)
