"""Rasch model (Rasch, 1960)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, Empirical
from scipy.special import expit

tf.flags.DEFINE_integer("nsubj", default=200, help="")
tf.flags.DEFINE_integer("nitem", default=25, help="")
tf.flags.DEFINE_integer("T", default=5000, help="Number of posterior samples.")

FLAGS = tf.flags.FLAGS


def main(_):
  # DATA
  trait_true = np.random.normal(size=[FLAGS.nsubj, 1])
  thresh_true = np.random.normal(size=[1, FLAGS.nitem])
  X_data = np.random.binomial(1, expit(trait_true - thresh_true))

  # MODEL
  trait = Normal(loc=0.0, scale=1.0, sample_shape=[FLAGS.nsubj, 1])
  thresh = Normal(loc=0.0, scale=1.0, sample_shape=[1, FLAGS.nitem])
  X = Bernoulli(logits=trait - thresh)

  # INFERENCE
  q_trait = Empirical(params=tf.get_variable("q_trait/params",
                                             [FLAGS.T, FLAGS.nsubj, 1]))
  q_thresh = Empirical(params=tf.get_variable("q_thresh/params",
                                              [FLAGS.T, 1, FLAGS.nitem]))

  inference = ed.HMC({trait: q_trait, thresh: q_thresh}, data={X: X_data})
  inference.run(step_size=0.1)

  # Alternatively, use variational inference.
  # q_trait = Normal(
  #     loc=tf.get_variable("q_trait/loc", [FLAGS.nsubj, 1]),
  #     scale=tf.nn.softplus(
  #         tf.get_variable("q_trait/scale", [FLAGS.nsubj, 1])))
  # q_thresh = Normal(
  #     loc=tf.get_variable("q_thresh/loc", [1, FLAGS.nitem]),
  #     scale=tf.nn.softplus(
  #         tf.get_variable("q_thresh/scale", [1, FLAGS.nitem])))

  # inference = ed.KLqp({trait: q_trait, thresh: q_thresh}, data={X: X_data})
  # inference.run(n_iter=2500, n_samples=10)

  # CRITICISM
  # Check that the inferred posterior mean captures the true traits.
  plt.scatter(trait_true, q_trait.mean().eval())
  plt.show()

  print("MSE between true traits and inferred posterior mean:")
  print(np.mean(np.square(trait_true - q_trait.mean().eval())))

if __name__ == "__main__":
  tf.app.run()
