"""Normal-normal model using Hamiltonian Monte Carlo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Empirical, Normal


def main(_):
  ed.set_seed(42)

  # DATA
  x_data = np.array([0.0] * 50)

  # MODEL: Normal-Normal with known variance
  mu = Normal(loc=0.0, scale=1.0)
  x = Normal(loc=mu, scale=1.0, sample_shape=50)

  # INFERENCE
  qmu = Empirical(params=tf.get_variable("qmu/params", [1000],
                                         initializer=tf.zeros_initializer()))

  # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
  inference = ed.HMC({mu: qmu}, data={x: x_data})
  inference.run()

  # CRITICISM
  sess = ed.get_session()
  mean, stddev = sess.run([qmu.mean(), qmu.stddev()])
  print("Inferred posterior mean:")
  print(mean)
  print("Inferred posterior stddev:")
  print(stddev)

  # Check convergence with visual diagnostics.
  samples = sess.run(qmu.params)

  # Plot histogram.
  plt.hist(samples, bins='auto')
  plt.show()

  # Trace plot.
  plt.plot(samples)
  plt.show()

if __name__ == "__main__":
  tf.app.run()
