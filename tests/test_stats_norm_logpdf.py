from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import norm
from scipy import stats

sess = tf.InteractiveSession()

print("Input: scalar")
x = tf.constant(0.0)
print(norm.logpdf(x).eval())
print(norm.logpdf(x, tf.constant(0.0), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant([1.0])).eval())
print(stats.norm.logpdf(0.0))
print()
x = tf.constant(0.623)
print(norm.logpdf(x).eval())
print(norm.logpdf(x, tf.constant(0.0), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant([1.0])).eval())
print(stats.norm.logpdf(0.623))
print()
print("Input: 1-dimensional vector")
x = tf.constant([0.0])
print(norm.logpdf(x).eval())
print(norm.logpdf(x, tf.constant(0.0), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant(0.0), tf.constant([1.0])).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant([1.0])).eval())
print(stats.norm.logpdf([0.0]))
print()
print("Input: 1x1 matrix")
x = tf.constant([[0.0]])
print(norm.logpdf(x).eval())
print(norm.logpdf(x, tf.constant(0.0), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant(1.0)).eval())
print(norm.logpdf(x, tf.constant(0.0), tf.constant([1.0])).eval())
print(norm.logpdf(x, tf.constant([0.0]), tf.constant([1.0])).eval())
print(stats.norm.logpdf([[0.0]]))
