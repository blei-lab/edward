from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.models import Normal
from vi import VariationalInference


# probability model: Normal-Normal with known variance
mu = tf.constant([0.0])
sigma = tf.constant([1.0])
pmu = Normal([mu, sigma])
x = Normal([pmu, sigma],
           lambda cond_set: tf.pack([cond_set[0] for n in range(50)]))

# variational model
mu2 = tf.Variable(tf.random_normal([1]))
sigma2 = tf.nn.softplus(tf.Variable(tf.random_normal([1])))
qmu = Normal([mu2, sigma2])

# inference
# analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
data = {x: np.array([0.0]*50, dtype=np.float32)}
inference = VariationalInference({pmu: qmu}, data)
inference.initialize()

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for t in range(10000):
    _, loss = sess.run([inference.train, inference.loss])
    if t % 100 == 0:
        print("iter: {:d}, loss: {:0.3f}".format(t, loss))
        print(sess.run([mu2, sigma2]))
