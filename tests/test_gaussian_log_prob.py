import numpy as np
import tensorflow as tf

from blackbox.dists import gaussian_log_prob
from scipy.stats import norm, multivariate_normal

#with tf.Session() as sess:
sess = tf.InteractiveSession()

print "Input: One-dimensional scalar"
x = tf.constant(0.0)
print gaussian_log_prob(x).eval()
print gaussian_log_prob(x, tf.zeros([1]), tf.constant(1.0)).eval()
print gaussian_log_prob(x, tf.zeros([1]), tf.ones([1])).eval()
print gaussian_log_prob(x, tf.zeros([1]), tf.diag(tf.ones([1]))).eval()
print norm.logpdf(0.0)
print
print "Input: One-dimensional vector"
x = tf.zeros([1])
print gaussian_log_prob(x).eval()
print norm.logpdf(0.0)
print
print "Input: Multi-dimensional vector"
x = tf.zeros([2])
print gaussian_log_prob(x).eval()
print gaussian_log_prob(x, tf.zeros([2]), tf.ones([2])).eval()
print gaussian_log_prob(x, tf.zeros([2]), tf.diag(tf.ones([2]))).eval()
print multivariate_normal.logpdf(np.zeros(2), np.zeros(2), np.diag(np.ones(2)))
print
x = tf.zeros([2])
print gaussian_log_prob(x, tf.zeros([2]),
                        tf.constant([[2.0, 0.5], [0.5, 1.0]])).eval()
print multivariate_normal.logpdf(np.zeros(2), np.zeros(2),
                                 np.array([[2.0, 0.5], [0.5, 1.0]]))

"""
print "Input: Multiple one-dimensional scalars"

print "Input: Multiple one-dimensional vectors"
x = tf.zeros([3])
print gaussian_log_prob(x, tf.zeros([1]), tf.diag(tf.ones([1]))).eval()

print "Input: Multiple multi-dimensional vectors"
x = tf.zeros([10, 2])
print gaussian_log_prob(x).eval()
print gaussian_log_prob(x, tf.zeros([2]), tf.diag(tf.ones([2]))).eval()
"""
