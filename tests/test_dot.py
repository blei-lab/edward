from __future__ import print_function
import numpy as np
import tensorflow as tf

from blackbox.util import dot

sess = tf.InteractiveSession()

a = tf.ones([5]) * np.arange(5)
b = tf.diag(tf.ones([5]))

print("a")
print(a.eval())
print()
print("b")
print(b.eval())
print()
print("dot(a, b)")
print(dot(b, a).eval())
print()
print("dot(b, a)")
print(dot(a, b).eval())
