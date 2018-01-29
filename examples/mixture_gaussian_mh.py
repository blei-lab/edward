"""Mixture of Gaussians.

Perform inference with Metropolis-Hastings. It utterly fails. This is
because we are proposing a sample in a high-dimensional space. The
acceptance ratio is so small that it is unlikely we'll ever accept a
proposed sample. A Gibbs-like extension ("MH within Gibbs"), which
does a separate MH in each dimension, may succeed.

References
----------
http://edwardlib.org/tutorials/unsupervised
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import six
import tensorflow as tf

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma, Normal)
from scipy.stats import norm

tf.flags.DEFINE_integer("N", default=500, help="Number of data points.")
tf.flags.DEFINE_integer("K", default=2, help="Number of components.")
tf.flags.DEFINE_integer("D", default=2, help="Dimensionality of data.")
tf.flags.DEFINE_integer("T", default=5000, help="Number of posterior samples.")

FLAGS = tf.flags.FLAGS


def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2))
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x


def main(_):
  ed.set_seed(42)

  # DATA
  x_data = build_toy_dataset(FLAGS.N)

  # MODEL
  pi = Dirichlet(concentration=tf.ones(FLAGS.K))
  mu = Normal(0.0, 1.0, sample_shape=[FLAGS.K, FLAGS.D])
  sigma = InverseGamma(concentration=1.0, rate=1.0,
                       sample_shape=[FLAGS.K, FLAGS.D])
  c = Categorical(logits=tf.log(pi) - tf.log(1.0 - pi), sample_shape=FLAGS.N)
  x = Normal(loc=tf.gather(mu, c), scale=tf.gather(sigma, c))

  # INFERENCE
  qpi = Empirical(params=tf.get_variable(
      "qpi/params",
      [FLAGS.T, FLAGS.K],
      initializer=tf.constant_initializer(1.0 / FLAGS.K)))
  qmu = Empirical(params=tf.get_variable("qmu/params",
                                         [FLAGS.T, FLAGS.K, FLAGS.D],
                                         initializer=tf.zeros_initializer()))
  qsigma = Empirical(params=tf.get_variable("qsigma/params",
                                            [FLAGS.T, FLAGS.K, FLAGS.D],
                                            initializer=tf.ones_initializer()))
  qc = Empirical(params=tf.get_variable("qc/params",
                                        [FLAGS.T, FLAGS.N],
                                        initializer=tf.zeros_initializer(),
                                        dtype=tf.int32))

  gpi = Dirichlet(concentration=tf.constant([1.4, 1.6]))
  gmu = Normal(loc=tf.constant([[1.0, 1.0], [-1.0, -1.0]]),
               scale=tf.constant([[0.5, 0.5], [0.5, 0.5]]))
  gsigma = InverseGamma(concentration=tf.constant([[1.1, 1.1], [1.1, 1.1]]),
                        rate=tf.constant([[1.0, 1.0], [1.0, 1.0]]))
  gc = Categorical(logits=tf.zeros([FLAGS.N, FLAGS.K]))

  inference = ed.MetropolisHastings(
      latent_vars={pi: qpi, mu: qmu, sigma: qsigma, c: qc},
      proposal_vars={pi: gpi, mu: gmu, sigma: gsigma, c: gc},
      data={x: x_data})

  inference.initialize()

  sess = ed.get_session()
  tf.global_variables_initializer().run()

  for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    t = info_dict['t']
    if t == 1 or t % inference.n_print == 0:
      qpi_mean, qmu_mean = sess.run([qpi.mean(), qmu.mean()])
      print("")
      print("Inferred membership probabilities:")
      print(qpi_mean)
      print("Inferred cluster means:")
      print(qmu_mean)

if __name__ == "__main__":
  tf.app.run()
