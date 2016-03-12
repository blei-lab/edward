from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import norm
from scipy import stats

sess = tf.InteractiveSession()

print("Input: None")
print(norm.entropy().eval())
print(stats.norm.entropy())
print()
print("Input: scalar")
scale = tf.constant(1.0)
print(norm.entropy(scale=scale).eval())
print(stats.norm.entropy(1.0))
print()
print("Input: 1-dimensional vector")
scale = tf.constant([1.0])
print(norm.entropy(scale=scale).eval())
print(stats.norm.entropy([1.0]))
