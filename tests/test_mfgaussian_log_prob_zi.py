from __future__ import print_function
import numpy as np
import tensorflow as tf
import blackbox as bb

from scipy import stats

sess = tf.InteractiveSession()

variational = bb.MFGaussian(1)
variational.m_unconst = tf.constant([0.0])
variational.s_unconst = tf.constant([0.0])

#print(variational.log_prob_zi(0, tf.constant(0.0, dtype=tf.float32)))
print("n_minibatch x d dimensional array of zs")
print("n_minibatch = 1, d = 1, q(z_{1,:}):")
print(variational.log_prob_zi(0,
          tf.constant([[0.0]], dtype=tf.float32)).eval())
print("n_minibatch = 1, d = 2, q(z_{1,:}):")
print(variational.log_prob_zi(0,
          tf.constant([[0.0, 0.0]], dtype=tf.float32)).eval())

print("n_minibatch = 2, d = 1, q(z_{1,:}):")
print(variational.log_prob_zi(0,
          tf.constant([[0.0], [0.0]], dtype=tf.float32)).eval())

print("n_minibatch = 2, d = 2, q(z_{1,:}):")
print(variational.log_prob_zi(0,
          tf.constant([[0.0, 0.0], [0.0, 0.0]], dtype=tf.float32)).eval())
