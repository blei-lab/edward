from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import bernoulli
from scipy import stats

sess = tf.InteractiveSession()

print(bernoulli.logpmf(tf.constant(0.0), tf.constant(0.5)).eval())
print(bernoulli.logpmf(tf.constant(0.0), tf.constant(0.5)).eval())
print(bernoulli.logpmf(tf.constant([0.0]), tf.constant(0.5)).eval())
print(bernoulli.logpmf(tf.constant([0.0]), tf.constant([0.5])).eval())
print(stats.bernoulli.logpmf(0.0, 0.5))
print()
print(bernoulli.logpmf(tf.constant(0.0), tf.constant(0.75)).eval())
print(stats.bernoulli.logpmf(0.0, 0.75))
