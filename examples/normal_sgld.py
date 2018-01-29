"""Correlated normal posterior. Inference with stochastic gradient
Langevin dynamics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Empirical, MultivariateNormalTriL


def main(_):
  ed.set_seed(42)

  # MODEL
  z = MultivariateNormalTriL(
      loc=tf.ones(2),
      scale_tril=tf.cholesky(tf.constant([[1.0, 0.8], [0.8, 1.0]])))

  # INFERENCE
  qz = Empirical(params=tf.get_variable("qz/params", [2000, 2]))

  inference = ed.SGLD({z: qz})
  inference.run(step_size=5.0)

  # CRITICISM
  sess = ed.get_session()
  mean, stddev = sess.run([qz.mean(), qz.stddev()])
  print("Inferred posterior mean:")
  print(mean)
  print("Inferred posterior stddev:")
  print(stddev)

if __name__ == "__main__":
  tf.app.run()
