from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import norm
from scipy import stats

sess = tf.InteractiveSession()

print("Input: One-dimensional scalar")
x = tf.constant(0.0)
print(norm.logpdf(x).eval())
print(norm.logpdf(x, tf.zeros([1]), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.zeros([1]), tf.ones([1])).eval())
print(norm.logpdf(x, tf.zeros([1]), tf.diag(tf.ones([1]))).eval())
print(stats.norm.logpdf(0.0))
print()
print("Input: One-dimensional vector")
x = tf.zeros([1])
print(norm.logpdf(x).eval())
print(stats.norm.logpdf(0.0))
print()
print("Input: Multi-dimensional vector")
x = tf.zeros([2])
print(norm.logpdf(x).eval())
print(norm.logpdf(x, tf.zeros([2]), tf.ones([2])).eval())
print(norm.logpdf(x, tf.zeros([2]), tf.diag(tf.ones([2]))).eval())
print(stats.multivariate_normal.logpdf(np.zeros(2), np.zeros(2), np.diag(np.ones(2))))
print()
x = tf.zeros([2])
print(norm.logpdf(x, tf.zeros([2]),
                        tf.constant([[2.0, 0.5], [0.5, 1.0]])).eval())
print(stats.multivariate_normal.logpdf(np.zeros(2), np.zeros(2),
                                 np.array([[2.0, 0.5], [0.5, 1.0]])))

"""
print("Input: Multiple one-dimensional scalars")

print("Input: Multiple one-dimensional vectors")
x = tf.zeros([3])
print(norm.logpdf(x, tf.zeros([1]), tf.diag(tf.ones([1]))).eval())

print("Input: Multiple multi-dimensional vectors")
x = tf.zeros([10, 2])
print(norm.logpdf(x).eval())
print(norm.logpdf(x, tf.zeros([2]), tf.diag(tf.ones([2]))).eval())
"""
