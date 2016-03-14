from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import bernoulli
from scipy import stats

sess = tf.InteractiveSession()

print("Input: scalar, int")
x = tf.constant(0)
print(bernoulli.logpmf(x, tf.constant(0.5)).eval())
print(bernoulli.logpmf(x, tf.constant([0.5])).eval())
print(stats.bernoulli.logpmf(0, 0.5))
print()
x = tf.constant(1)
print(bernoulli.logpmf(x, tf.constant(0.75)).eval())
print(stats.bernoulli.logpmf(1, 0.75))
print()
print("Input: scalar, float")
x = tf.constant(0.0)
print(bernoulli.logpmf(x, tf.constant(0.5)).eval())
print(bernoulli.logpmf(x, tf.constant([0.5])).eval())
print(stats.bernoulli.logpmf(0.0, 0.5))
print()
print("Input: 1-dimensional vector, int")
x = tf.constant([0])
print(bernoulli.logpmf(x, tf.constant(0.5)).eval())
print(bernoulli.logpmf(x, tf.constant([0.5])).eval())
print(stats.bernoulli.logpmf([0], 0.5))
print()
print("Input: 1-dimensional vector, float")
x = tf.constant([0.0])
print(bernoulli.logpmf(x, tf.constant(0.5)).eval())
print(bernoulli.logpmf(x, tf.constant([0.5])).eval())
print(stats.bernoulli.logpmf([0.0], 0.5))
