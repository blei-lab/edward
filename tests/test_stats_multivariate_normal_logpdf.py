from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.stats import multivariate_normal
from scipy import stats

sess = tf.InteractiveSession()

print("Input: 2-dimensional vector, float")
x = tf.constant([0.0, 0.0])
print(multivariate_normal.logpdf(x).eval())
print(multivariate_normal.logpdf(x, tf.zeros([2]), tf.ones([2])).eval())
print(multivariate_normal.logpdf(x, tf.zeros([2]), tf.diag(tf.ones([2]))).eval())
print(stats.multivariate_normal.logpdf(np.zeros(2), np.zeros(2), np.diag(np.ones(2))))
print()
x = tf.constant([0.0, 0.0])
print(multivariate_normal.logpdf(x,
        tf.zeros([2]),
        tf.constant([[2.0, 0.5], [0.5, 1.0]])).eval())
print(stats.multivariate_normal.logpdf(np.zeros(2),
        np.zeros(2),
        np.array([[2.0, 0.5], [0.5, 1.0]])))
print()
print("Input: 2-dimensional vector, int")
x = tf.constant([0, 0])
print(multivariate_normal.logpdf(x).eval())
print(multivariate_normal.logpdf(x, tf.zeros([2]), tf.ones([2])).eval())
print(multivariate_normal.logpdf(x, tf.zeros([2]), tf.diag(tf.ones([2]))).eval())
print(stats.multivariate_normal.logpdf(np.zeros(2), np.zeros(2), np.diag(np.ones(2))))
