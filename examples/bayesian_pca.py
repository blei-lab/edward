from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import Normal, Gamma
from edward.stats import norm

import scipy
from scipy.stats import gaussian_kde

ed.set_seed(142)

# Simulate data
N = 5000
D = 2
K = 1

sigma = 1

x_train = np.zeros((D, N))

w = norm.rvs(loc=0, scale=2, size=(D, K))
z = norm.rvs(loc=0, scale=1, size=(K, N))

mean = np.dot(w, z)

for d in range(D):
  for n in range(N):
    x_train[d, n] = norm.rvs(loc=mean[d, n], scale=sigma)

print(w)

# PPCA Model

w = Normal(mu=tf.zeros([D, K]), sigma=10.0*tf.ones([D, K]))
z = Normal(mu=tf.zeros([K, N]), sigma=1.0*tf.ones([K, N]))

x = Normal(mu=tf.matmul(w, z), sigma=1.0)


# Define Variational Approximation Distributions

qw = Normal(mu=tf.Variable(tf.random_normal([D, K])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qz = Normal(mu=tf.Variable(tf.random_normal([K, N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([K, N]))))

inference = ed.KLqp({w: qw, z: qz}, data={x: x_train})

# Run Variational Inference

sess = ed.get_session()
init = tf.initialize_all_variables()
inference.run(n_iter=500, n_print=100, n_samples=10)

print(sess.run(qw.mean()))


# Subsampling

M = 100

w = Normal(mu=tf.zeros([D, K]), sigma=10.0*tf.ones([D, K]))
z = Normal(mu=tf.zeros([K, M]), sigma=1.0*tf.ones([K, M]))

x = Normal(mu=tf.matmul(w, z), sigma=1.0)

qw_variables = [tf.Variable(tf.random_normal([D, K])),
                tf.Variable(tf.random_normal([D, K]))]
qw = Normal(mu=qw_variables[0], sigma=tf.nn.softplus(qw_variables[1]))

qz_variables = [tf.Variable(tf.random_normal([K, M])),
                tf.Variable(tf.random_normal([K, M]))]
qz = Normal(mu=qz_variables[0], sigma=tf.nn.softplus(qz_variables[1]))

x_ph = tf.placeholder(tf.float32, [D, M])
inference = ed.KLqp({w: qw, z: qz}, data={x: x_ph})

optimizer = tf.train.RMSPropOptimizer(learning_rate=1.0, decay=0.9)

inference.initialize(n_samples=10,
                     optimizer=optimizer,
                     scale={x: float(N) / M, z: float(N) / M})

init = tf.initialize_all_variables()
init.run()


def next_batch(M):
  return x_train[:, np.random.choice(N, M)]

init_local = tf.initialize_variables(qz_variables)

sess = ed.get_session()
for i in range(250):
  x_batch = next_batch(M)
  for _ in range(25):
    inference.update(feed_dict={x_ph: x_batch})
  init_local.run()

print(sess.run(qw.mean()))
