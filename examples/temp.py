#!/usr/bin/env python
"""Generative adversarial network for toy Gaussian data
(Goodfellow et al., 2014).

Inspired by a blog post by Eric Jang.

Note there are several common failure modes, such as
(1) saturation of either discriminative or generative network;
(2) the generator running into a local optima that produces a Gaussian
somewhere around -1 rather than at the true data; and
(3) mode collapse around the true Gaussian, where the variance is
severely underestimated.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from matplotlib import animation
from scipy.stats import norm


def next_batch(N):
  samples = np.random.normal(4.0, 0.5, N)
  samples.sort()
  return samples


def fully_connected(x, output_dim, activation_fn, scope=None):
  with tf.variable_scope(scope):
    w = tf.get_variable("w", [x.get_shape()[1], output_dim],
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", [output_dim],
                        initializer=tf.constant_initializer(0.0))
    if activation_fn:
      return activation_fn(tf.matmul(x, w) + b)
    else:
      return tf.matmul(x, w) + b


def generative_network(z):
  h0 = fully_connected(z, hidden_size, tf.nn.relu, "G0")
  h1 = fully_connected(h0, 1, None, "G1")
  return h1


def discriminative_network(x):
  """Outputs probability in logits."""
  h0 = fully_connected(x, hidden_size * 2, tf.tanh, "D0")
  h1 = fully_connected(h0, hidden_size * 2, tf.tanh, "D1")
  h2 = fully_connected(h1, hidden_size * 2, tf.tanh, "D2")
  h3 = fully_connected(h2, 1, None, "D3")
  return h3


def get_samples(num_points=10000, num_bins=100):
  """Return a tuple (db, pd, pg), where
  + db is the discriminator's decision boundary;
  + pd is a histogram of samples from the data distribution;
  + pg is a histogram of samples from the generative model.
  """
  sess = ed.get_session()
  bins = np.linspace(-toy_range, toy_range, num_bins)

  # Decision boundary
  with tf.variable_scope("Disc", reuse=True):
    p_true = discriminative_network(x_ph)

  xs = np.linspace(-toy_range, toy_range, num_points)
  db = np.zeros((num_points, 1))
  for i in range(num_points // batch_size):
    db[batch_size * i:batch_size * (i + 1)] = sess.run(p_true, {
        x_ph: np.reshape(
            xs[batch_size * i:batch_size * (i + 1)],
            (batch_size, 1)
        )
    })

  # Data samples
  d = next_batch(num_points)
  pd, _ = np.histogram(d, bins=bins, density=True)

  # Generated samples
  z_ph = tf.placeholder(tf.float32, [batch_size, 1])
  with tf.variable_scope("Gen", reuse=True):
    G = generative_network(z_ph)

  zs = np.linspace(-toy_range, toy_range, num_points)
  g = np.zeros((num_points, 1))
  for i in range(num_points // batch_size):
    g[batch_size * i:batch_size * (i + 1)] = sess.run(G, {
        z_ph: np.reshape(
            zs[batch_size * i:batch_size * (i + 1)],
            (batch_size, 1)
        )
    })
  pg, _ = np.histogram(g, bins=bins, density=True)

  return db, pd, pg


sns.set(color_codes=True)
ed.set_seed(42)

toy_range = 8  # range of data
batch_size = 12  # batch size during training
hidden_size = 4  # number of hidden units
n_iter = 1000  # number of training iterations
n_print = 10  # print every number of iterations

anim_frames = []
anim_path = None  # file path of outputted animation

# DATA. We use a placeholder to represent a minibatch. During
# inference, we generate data on the fly and feed ``x_ph``.
x_ph = tf.placeholder(tf.float32, [batch_size, 1])

# MODEL
stop = tf.cast(toy_range, tf.float32)
z = tf.linspace(-stop, stop, batch_size) + 0.01 * tf.random_normal([batch_size])
z = tf.reshape(z, [batch_size, 1])
with tf.variable_scope("Gen"):
  x = generative_network(z)

# INFERENCE
initial_learning_rate = 0.03
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    initial_learning_rate,
    global_step,
    decay_steps=150,
    decay_rate=0.95)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer_d = tf.train.GradientDescentOptimizer(learning_rate)

# TODO
# + there's some randomness still in this example.
#   i think the loss values are always the same though..
#   maybe it's in the graph construction or whichever updates go first in the session runs?
inference = ed.GANInference(
    data={x: x_ph}, discriminator=discriminative_network)
inference.initialize(
    optimizer=optimizer, optimizer_d=optimizer,
    global_step=global_step, global_step_d=global_step,
    n_iter=n_iter, n_print=n_print)
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  x_data = next_batch(batch_size).reshape([batch_size, 1])
  info_dict = inference.update(feed_dict={x_ph: x_data})
  inference.print_progress(info_dict)

# CRITICISM
db, pd, pg = get_samples()
db_x = np.linspace(-toy_range, toy_range, len(db))
p_x = np.linspace(-toy_range, toy_range, len(pd))
f, ax = plt.subplots(1)
ax.plot(db_x, db, label="Decision boundary")
ax.set_ylim(0, 1)
plt.plot(p_x, pd, label="Real data")
plt.plot(p_x, pg, label="Generated data")
plt.title("1D Generative Adversarial Network")
plt.xlabel("Data values")
plt.ylabel("Probability density")
plt.legend()
plt.show()
