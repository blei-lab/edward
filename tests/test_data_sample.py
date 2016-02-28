from __future__ import print_function
import blackbox as bb
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

print("tf.Tensor data type, single sample")
data = bb.Data(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32),
                n_minibatch=1)
for t in range(20):
    if t == 10:
        print()

    x = data.sample()
    print(x.eval())

print(type(x))

print("tf.Tensor data type, multiple samples")
data = bb.Data(tf.constant((0, 1, 0, 0, 0, 0, 0, 0, 0, 1), dtype=tf.float32),
                n_minibatch=2)
for t in range(10):
    if t == 5:
        print()

    x = data.sample()
    print(x.eval())

print(type(x))

print("np.ndarray data type, single sample")
data = bb.Data(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
               n_minibatch=1)
for t in range(20):
    if t == 10:
        print()

    x = data.sample()
    print(x)

print(type(x))

print("np.ndarray data type, multiple samples")
data = bb.Data(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
               n_minibatch=2)
for t in range(10):
    if t == 5:
        print()

    x = data.sample()
    print(x)

print(type(x))

print("dict data type, single sample")
data = bb.Data(dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
               n_minibatch=1)
for t in range(20):
    if t == 10:
        print()

    x = data.sample()
    print(x)

print(type(x))

print("dict data type, multiple samples")
data = bb.Data(dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
               n_minibatch=2)
for t in range(10):
    if t == 5:
        print()

    x = data.sample()
    print(x)

print(type(x))
