"""Stochastic recursion.

We sample from a geometric random variable by using samples from
Bernoulli random variable. It uses a recursive function and requires
lazy evaluation of the condition.

Recursion is not available in TensorFlow and so neither is stochastic
recursion available in Edward's modeling language. There are several
alternatives: (stochastic) while loops, wrapping around a Python
implementation (`tf.py_func`), and a CPS-style formulation.

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
    cond = tf.cast(Bernoulli(probs=p), tf.bool)

    def fn1():
      return tf.constant(0)

    def fn2():
      return geometric(p) + 1

    # TensorFlow builds the op non-lazily, unrolling both functions
    # before it checks the condition. This makes this function fail.
    return tf.cond(cond, fn1, fn2)


def main(_):
  p = tf.constant(0.9)
  geom = geometric(p)

  sess = tf.Session()
  samples = [sess.run(geom) for _ in range(1000)]
  plt.hist(samples, bins='auto')

if __name__ == "__main__":
  tf.app.run()
