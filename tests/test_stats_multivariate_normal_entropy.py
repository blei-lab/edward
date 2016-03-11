from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import multivariate_normal
from scipy import stats

sess = tf.InteractiveSession()

print("Input: None")
print(multivariate_normal.entropy().eval())
print(stats.multivariate_normal.entropy())
print()
print("Input: 2-dimensional vector")
cov = tf.constant([1.0, 1.0])
print(multivariate_normal.entropy(cov=cov).eval())
print(2.83788)
print()
print("Input: 2x2 matrix")
cov = tf.constant([[1.0, 0.0], [0.0, 1.0]])
print(multivariate_normal.entropy(cov=cov).eval())
print(2.83788)
