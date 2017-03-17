import edward as ed
from edward.models import Normal
import numpy as np
import tensorflow as tf


def build_toy_dataset(U, V, N, M, noise_std=0.1):
  R = np.dot(np.transpose(U), V) + np.random.normal(0, noise_std, size=(N, M))
  return R


N = 30  # number of users
M = 20  # number of movies
D = 3  # number of latent factors

# true latent factors
U_true = np.random.randn(D, N).astype(np.float32)
V_true = np.random.randn(D, M).astype(np.float32)

# DATA
R_train = build_toy_dataset(U_true, V_true, N, M)
R_test = build_toy_dataset(U_true, V_true, N, M)

# MODEL
U = Normal(mu=tf.zeros([D, N]), sigma=tf.ones([D, N]))
V = Normal(mu=tf.zeros([D, M]), sigma=tf.ones([D, M]))
R = Normal(mu=tf.matmul(tf.transpose(U), V), sigma=tf.ones([N, M]))

# INFERENCE
qU = Normal(mu=tf.Variable(tf.random_normal([D, N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, N]))))
qV = Normal(mu=tf.Variable(tf.random_normal([D, M])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, M]))))

inference = ed.KLqp({U: qU, V: qV}, data={R: R_train})
inference.run()

# CRITICISM
qR = Normal(mu=tf.matmul(tf.transpose(qU), qV), sigma=tf.ones([N, M]))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={qR: R_test}))
