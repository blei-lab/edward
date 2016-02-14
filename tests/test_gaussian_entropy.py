from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import gaussian_entropy
from scipy.stats import norm, multivariate_normal

#with tf.Session() as sess:
sess = tf.InteractiveSession()

print("Input: One-dimensional scalar")
x = tf.constant(1.0)
print(gaussian_entropy(x).eval())
print(1.41894)
print()
print("Input: One-dimensional vector")
x = tf.ones([1])
print(gaussian_entropy(x).eval())
print(1.41894)
print()
print("Input: Multi-dimensional vector")
x = tf.ones([2])
print(gaussian_entropy(x).eval())
print(2.83788)
print()
