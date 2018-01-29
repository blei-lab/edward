"""Probabilistic principal components analysis (Tipping and Bishop, 1999).

Inference uses data subsampling.

References
----------
http://edwardlib.org/tutorials/probabilistic-pca
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal

tf.flags.DEFINE_integer("N", default=5000, help="Number of data points.")
tf.flags.DEFINE_integer("M", default=100, help="Batch size during training.")
tf.flags.DEFINE_integer("D", default=2, help="Data dimensionality.")
tf.flags.DEFINE_integer("K", default=1, help="Latent dimensionality.")

FLAGS = tf.flags.FLAGS


def build_toy_dataset(N, D, K, sigma=1):
  x_train = np.zeros((D, N))
  w = np.random.normal(0.0, 2.0, size=(D, K))
  z = np.random.normal(0.0, 1.0, size=(K, N))
  mean = np.dot(w, z)
  for d in range(D):
    for n in range(N):
      x_train[d, n] = np.random.normal(mean[d, n], sigma)

  print("True principal axes:")
  print(w)
  return x_train


def next_batch(x_train, M):
  idx_batch = np.random.choice(FLAGS.N, M)
  return x_train[:, idx_batch], idx_batch


def main(_):
  ed.set_seed(142)

  # DATA
  x_train = build_toy_dataset(FLAGS.N, FLAGS.D, FLAGS.K)

  # MODEL
  w = Normal(loc=0.0, scale=10.0, sample_shape=[FLAGS.D, FLAGS.K])
  z = Normal(loc=0.0, scale=1.0, sample_shape=[FLAGS.M, FLAGS.K])
  x = Normal(loc=tf.matmul(w, z, transpose_b=True),
             scale=tf.ones([FLAGS.D, FLAGS.M]))

  # INFERENCE
  qw_variables = [tf.get_variable("qw/loc", [FLAGS.D, FLAGS.K]),
                  tf.get_variable("qw/scale", [FLAGS.D, FLAGS.K])]
  qw = Normal(loc=qw_variables[0], scale=tf.nn.softplus(qw_variables[1]))

  qz_variables = [tf.get_variable("qz/loc", [FLAGS.N, FLAGS.K]),
                  tf.get_variable("qz/scale", [FLAGS.N, FLAGS.K])]
  idx_ph = tf.placeholder(tf.int32, FLAGS.M)
  qz = Normal(loc=tf.gather(qz_variables[0], idx_ph),
              scale=tf.nn.softplus(tf.gather(qz_variables[1], idx_ph)))

  x_ph = tf.placeholder(tf.float32, [FLAGS.D, FLAGS.M])
  inference_w = ed.KLqp({w: qw}, data={x: x_ph, z: qz})
  inference_z = ed.KLqp({z: qz}, data={x: x_ph, w: qw})

  scale_factor = float(FLAGS.N) / FLAGS.M
  inference_w.initialize(scale={x: scale_factor, z: scale_factor},
                         var_list=qz_variables,
                         n_samples=5)
  inference_z.initialize(scale={x: scale_factor, z: scale_factor},
                         var_list=qw_variables,
                         n_samples=5)

  sess = ed.get_session()
  tf.global_variables_initializer().run()
  for _ in range(inference_w.n_iter):
    x_batch, idx_batch = next_batch(x_train, FLAGS.M)
    for _ in range(5):
      inference_z.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})

    info_dict = inference_w.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})
    inference_w.print_progress(info_dict)

    t = info_dict['t']
    if t % 100 == 0:
      print("\nInferred principal axes:")
      print(sess.run(qw.mean()))

if __name__ == "__main__":
  tf.app.run()
