from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import bernoulli
from scipy import stats

sess = tf.InteractiveSession()

x = tf.constant(0.0)
print(bernoulli.logpmf(x, tf.constant(0.5)).eval())
print(stats.bernoulli.logpmf(0.0, 0.5))
print(bernoulli.logpmf(x, tf.constant([0.75])).eval())
print(stats.bernoulli.logpmf(0.0, 0.75))
