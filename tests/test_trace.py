import tensorflow as tf

from blackbox.util import trace

#sess = tf.Session()
sess = tf.InteractiveSession()

X = tf.diag([2])
print X.eval()
print trace(X).eval()

X = tf.diag(tf.ones([2]))
print X.eval()
print trace(X).eval()
