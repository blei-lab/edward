from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import beta
from scipy import stats

sess = tf.InteractiveSession()

print(beta.logpdf(tf.constant(0.5),
                  tf.constant(0.5), tf.constant(0.5)).eval())
print(beta.logpdf(tf.constant(0.5),
                  tf.constant([0.5]), tf.constant(0.5)).eval())
print(beta.logpdf(tf.constant(0.5),
                  tf.constant(0.5), tf.constant([0.5])).eval())
print(beta.logpdf(tf.constant([0.5]),
                  tf.constant(0.5), tf.constant([0.5])).eval())
print(stats.beta.logpdf(0.5, 0.5, 0.5))
print()
print(beta.logpdf(tf.constant(0.6),
                  tf.constant(0.5), tf.constant(0.5)).eval())
print(stats.beta.logpdf(0.6, 0.5, 0.5))
