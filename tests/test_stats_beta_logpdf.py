from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import beta
from scipy import stats

sess = tf.InteractiveSession()

print("Input: scalar")
x = tf.constant(0.5)
print(beta.logpdf(x, tf.constant(0.5), tf.constant(0.5)).eval())
print(beta.logpdf(x, tf.constant([0.5]), tf.constant(0.5)).eval())
print(beta.logpdf(x, tf.constant(0.5), tf.constant([0.5])).eval())
print(beta.logpdf(x, tf.constant([0.5]), tf.constant([0.5])).eval())
print(stats.beta.logpdf(0.5, 0.5, 0.5))
print()
x = tf.constant(0.6)
print(beta.logpdf(x, tf.constant(0.5), tf.constant(0.5)).eval())
print(stats.beta.logpdf(0.6, 0.5, 0.5))
print()
print("Input: 1-dimensional vector")
x = tf.constant([0.5])
print(beta.logpdf(x, tf.constant(0.5), tf.constant(0.5)).eval())
print(beta.logpdf(x, tf.constant([0.5]), tf.constant(0.5)).eval())
print(beta.logpdf(x, tf.constant(0.5), tf.constant([0.5])).eval())
print(beta.logpdf(x, tf.constant([0.5]), tf.constant([0.5])).eval())
print(stats.beta.logpdf([0.5], 0.5, 0.5))
