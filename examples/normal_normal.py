"""Normal-normal model using Hamiltonian Monte Carlo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal


def model():
  mu = Normal(loc=0.0, scale=1.0, name="mu")
  x = Normal(loc=mu, scale=1.0, sample_shape=50, name="x")
  return x


def main(_):
  tf.set_random_seed(42)

  x_data = np.array([0.0] * 50)

  # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
  qmu = tf.get_variable("qmu", [])
  new_state, kernel_results = ed.hmc(
      model,
      step_size=0.2,
      current_state=qmu,
      align_latent=lambda name: {"mu" : "qmu"}.get(name),
      align_data=lambda name: {"x": "x"}.get(name),
      x=x_data)

  qmu_update = qmu.assign(new_state)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  samples = []
  num_accept = 0
  for t in range(2500):
    sample, accept = sess.run([qmu_update, kernel_results.is_accepted])
    samples.append(sample)
    num_accept += float(accept)
    if t % 100 == 0:
      print("Step {}, Acceptance Rate {:.3}".format(t, num_accept / max(t, 1)))

  samples = samples[500:]

  mean = np.mean(samples)
  stddev = np.std(samples)
  print("Inferred posterior mean:")
  print(mean)
  print("Inferred posterior stddev:")
  print(stddev)

  # Plot histogram.
  plt.hist(samples, bins='auto')
  plt.show()

  # Trace plot.
  plt.plot(samples)
  plt.show()

if __name__ == "__main__":
  tf.app.run()
