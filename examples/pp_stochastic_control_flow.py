"""Stochastic control flow.

We sample from a geometric random variable by using samples from
Bernoulli random variables. It requires a while loop whose condition
is stochastic.

References
----------
https://probmods.org/chapters/02-generative-models.html#stochastic-recursion
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf

from edward.models import Bernoulli


def geometric(p):
  i = tf.constant(0)
  sample = tf.while_loop(
      cond=lambda i: tf.cast(1 - Bernoulli(probs=p), tf.bool),
      body=lambda i: i + 1,
      loop_vars=[i])
  return sample


def main(_):
  p = 0.1
  geom = geometric(p)

  sess = tf.Session()
  samples = [sess.run(geom) for _ in range(1000)]
  plt.hist(samples, bins='auto')
  plt.title("Geometric({0})".format(p))
  plt.show()

if __name__ == "__main__":
  tf.app.run()
