"""Dirichlet process.

We implement sample generation from a Dirichlet process (with no base
distribution) via its stick breaking construction. It is a streamlined
implementation of the `DirichletProcess` random variable in Edward.

References
----------
https://probmods.org/chapters/12-non-parametric-models.html#infinite-discrete-distributions-the-dirichlet-processes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf

from edward.models import Bernoulli, Beta, DirichletProcess, Exponential, Normal


def dirichlet_process(alpha):
  """Demo of stochastic while loop for stick breaking construction."""
  def cond(k, beta_k):
    # End while loop (return False) when flip is heads.
    flip = Bernoulli(beta_k)
    return tf.cast(1 - flip, tf.bool)

  def body(k, beta_k):
    beta_k = Beta(1.0, alpha)
    return k + 1, beta_k

  k = tf.constant(0)
  beta_k = Beta(1.0, alpha)
  stick_num, stick_beta = tf.while_loop(cond, body, loop_vars=[k, beta_k])
  return stick_num


def main(_):
  dp = dirichlet_process(10.0)

  # The number of sticks broken is dynamic, changing across evaluations.
  sess = tf.Session()
  print(sess.run(dp))
  print(sess.run(dp))

  # Demo of the DirichletProcess random variable in Edward.
  base = Normal(0.0, 1.0)

  # Highly concentrated DP.
  alpha = 1.0
  dp = DirichletProcess(alpha, base)
  x = dp.sample(1000)
  samples = sess.run(x)
  plt.hist(samples, bins=100, range=(-3.0, 3.0))
  plt.title("DP({0}, N(0, 1))".format(alpha))
  plt.show()

  # More spread out DP.
  alpha = 50.0
  dp = DirichletProcess(alpha, base)
  x = dp.sample(1000)
  samples = sess.run(x)
  plt.hist(samples, bins=100, range=(-3.0, 3.0))
  plt.title("DP({0}, N(0, 1))".format(alpha))
  plt.show()

  # States persist across calls to sample() in a DP.
  alpha = 1.0
  dp = DirichletProcess(alpha, base)
  x = dp.sample(50)
  y = dp.sample(75)
  samples_x, samples_y = sess.run([x, y])
  plt.subplot(211)
  plt.hist(samples_x, bins=100, range=(-3.0, 3.0))
  plt.title("DP({0}, N(0, 1)) across two calls to sample()".format(alpha))
  plt.subplot(212)
  plt.hist(samples_y, bins=100, range=(-3.0, 3.0))
  plt.show()

  # `theta` is the distribution indirectly returned by the DP.
  # Fetching theta is the same as fetching the Dirichlet process.
  dp = DirichletProcess(alpha, base)
  theta = Normal(0.0, 1.0, value=tf.cast(dp, tf.float32))
  print(sess.run([dp, theta]))
  print(sess.run([dp, theta]))

  # DirichletProcess can also take in non-scalar concentrations and bases.
  alpha = tf.constant([0.1, 0.6, 0.4])
  base = Exponential(rate=tf.ones([5, 2]))
  dp = DirichletProcess(alpha, base)
  print(dp)

if __name__ == "__main__":
  plt.style.use('ggplot')
  tf.app.run()
