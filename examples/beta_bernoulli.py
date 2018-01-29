"""A simple coin flipping example. Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta, Empirical


def main(_):
  ed.set_seed(42)

  # DATA
  x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

  # MODEL
  p = Beta(1.0, 1.0)
  x = Bernoulli(probs=p, sample_shape=10)

  # INFERENCE
  qp = Empirical(params=tf.get_variable(
      "qp/params", [1000], initializer=tf.constant_initializer(0.5)))

  proposal_p = Beta(3.0, 9.0)

  inference = ed.MetropolisHastings({p: qp}, {p: proposal_p}, data={x: x_data})
  inference.run()

  # CRITICISM
  # exact posterior has mean 0.25 and std 0.12
  sess = ed.get_session()
  mean, stddev = sess.run([qp.mean(), qp.stddev()])
  print("Inferred posterior mean:")
  print(mean)
  print("Inferred posterior stddev:")
  print(stddev)

  x_post = ed.copy(x, {p: qp})
  tx_rep, tx = ed.ppc(
      lambda xs, zs: tf.reduce_mean(tf.cast(xs[x_post], tf.float32)),
      data={x_post: x_data})
  ed.ppc_stat_hist_plot(
      tx[0], tx_rep, stat_name=r'$T \equiv$mean', bins=10)
  plt.show()

if __name__ == "__main__":
  tf.app.run()
