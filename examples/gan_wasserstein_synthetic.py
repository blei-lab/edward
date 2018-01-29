"""Wasserstein generative adversarial network for toy Gaussian data
(Arjovsky et al., 2017). A gradient penalty is used to approximate the
1-Lipschitz functional family in the Wasserstein distance (Gulrajani
et al., 2017).

Inspired by a blog post by Eric Jang.

Note there are several common failure modes, such as
(1) saturation of either discriminative or generative network;
(2) mode collapse around the true Gaussian, where the variance is
severely underestimated.

References
----------
http://edwardlib.org/tutorials/gan
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from scipy.stats import norm

tf.flags.DEFINE_integer("M", default=12, help="Batch size during training.")
tf.flags.DEFINE_integer("H", default=4, help="Number of hidden units.")

FLAGS = tf.flags.FLAGS


def next_batch(N):
  samples = np.random.normal(4.0, 0.5, N)
  samples.sort()
  return samples


def generative_network(eps):
  net = tf.layers.dense(eps, FLAGS.H, activation=tf.nn.relu)
  net = tf.layers.dense(net, 1, activation=None)
  return net


def discriminative_network(x):
  """Outputs probability in logits."""
  net = tf.layers.dense(x, FLAGS.H * 2, activation=tf.tanh)
  net = tf.layers.dense(net, FLAGS.H * 2, activation=tf.tanh)
  net = tf.layers.dense(net, FLAGS.H * 2, activation=tf.tanh)
  net = tf.layers.dense(net, 1, activation=None)
  return net


def get_samples(x_ph, num_points=10000, num_bins=100):
  """Return a tuple (db, pd, pg), where
  + db is the discriminator's decision boundary;
  + pd is a histogram of samples from the data distribution;
  + pg is a histogram of samples from the generative model.
  """
  sess = ed.get_session()
  bins = np.linspace(-8, 8, num_bins)

  # Decision boundary
  with tf.variable_scope("Disc", reuse=True):
    p_true = tf.sigmoid(discriminative_network(x_ph))

  xs = np.linspace(-8, 8, num_points)
  db = np.zeros((num_points, 1))
  for i in range(num_points // FLAGS.M):
    db[FLAGS.M * i:FLAGS.M * (i + 1)] = sess.run(
        p_true, {x_ph: np.reshape(xs[FLAGS.M * i:FLAGS.M * (i + 1)],
                                  (FLAGS.M, 1))})

  # Data samples
  d = next_batch(num_points)
  pd, _ = np.histogram(d, bins=bins, density=True)

  # Generated samples
  eps_ph = tf.placeholder(tf.float32, [FLAGS.M, 1])
  with tf.variable_scope("Gen", reuse=True):
    G = generative_network(eps_ph)

  epss = np.linspace(-8, 8, num_points)
  g = np.zeros((num_points, 1))
  for i in range(num_points // FLAGS.M):
    g[FLAGS.M * i:FLAGS.M * (i + 1)] = sess.run(
        G, {eps_ph: np.reshape(epss[FLAGS.M * i:FLAGS.M * (i + 1)],
                               (FLAGS.M, 1))})
  pg, _ = np.histogram(g, bins=bins, density=True)

  return db, pd, pg


def main(_):
  sns.set(color_codes=True)
  ed.set_seed(42)

  # DATA. We use a placeholder to represent a minibatch. During
  # inference, we generate data on the fly and feed `x_ph`.
  x_ph = tf.placeholder(tf.float32, [FLAGS.M, 1])

  # MODEL
  with tf.variable_scope("Gen"):
    eps = tf.linspace(-8.0, 8.0, FLAGS.M) + 0.01 * tf.random_normal([FLAGS.M])
    eps = tf.reshape(eps, [FLAGS.M, 1])
    x = generative_network(eps)

  # INFERENCE
  optimizer = tf.train.GradientDescentOptimizer(0.03)
  optimizer_d = tf.train.GradientDescentOptimizer(0.03)

  inference = ed.WGANInference(
      data={x: x_ph}, discriminator=discriminative_network)
  inference.initialize(
      optimizer=optimizer, optimizer_d=optimizer_d, penalty=0.1,
      n_iter=1000)
  tf.global_variables_initializer().run()

  for _ in range(inference.n_iter):
    x_data = next_batch(FLAGS.M).reshape([FLAGS.M, 1])
    for _ in range(5):
      info_dict_d = inference.update(feed_dict={x_ph: x_data}, variables="Disc")

    info_dict = inference.update(feed_dict={x_ph: x_data}, variables="Gen")
    info_dict['t'] = info_dict['t'] // 6  # say set of 6 updates is 1 iteration
    info_dict['loss_d'] = info_dict_d['loss_d']  # get disc loss from update
    inference.print_progress(info_dict)

  # CRITICISM
  db, pd, pg = get_samples(x_ph)
  db_x = np.linspace(-8, 8, len(db))
  p_x = np.linspace(-8, 8, len(pd))
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

if __name__ == "__main__":
  tf.app.run()
