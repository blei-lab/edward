"""A simple coin flipping example. Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta


def model():
  p = Beta(1.0, 1.0, name="p")
  x = Bernoulli(probs=p, sample_shape=10, name="x")
  return x


def proposal(p):
  proposal_p = Beta(3.0, 9.0, name="proposal/p")
  return proposal_p


def main(_):
  tf.set_random_seed(42)

  x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

  qp = tf.get_variable("qp", initializer=0.5)
  new_state, is_accepted, _, _ = ed.metropolis_hastings(
      model, proposal,
      current_state=qp,
      align_latent=lambda name: {"p": "qp"}.get(name),
      align_proposal=lambda name: {"p": "proposal/p"}.get(name),
      align_data=lambda name: {"x": "x_data"}.get(name),
      x_data=x_data)
  qp_update = qp.assign(new_state)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  samples = []
  num_accept = 0
  for t in range(2500):
    sample, accept = sess.run([qp_update, is_accepted])
    samples.append(sample)
    num_accept += float(accept)
    if t % 100 == 0:
      print("Step {}, Acceptance Rate {:.3}".format(t, num_accept / max(t, 1)))

  samples = samples[500:]

  # exact posterior has mean 0.25 and std 0.12
  mean = np.mean(samples)
  stddev = np.std(samples)
  print("Inferred posterior mean:")
  print(mean)
  print("Inferred posterior stddev:")
  print(stddev)

  # TODO
  # x_post = ed.copy(x, {p: qp})
  # tx_rep, tx = ed.ppc(
  #     lambda xs, zs: tf.reduce_mean(tf.cast(xs[x_post], tf.float32)),
  #     data={x_post: x_data})
  # ed.ppc_stat_hist_plot(
  #     tx[0], tx_rep, stat_name=r'$T \equiv$mean', bins=10)
  # plt.show()

if __name__ == "__main__":
  tf.app.run()
