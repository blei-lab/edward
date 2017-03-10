#!/usr/bin/env python
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
  sample = tf.while_loop(cond=lambda i: tf.cast(Bernoulli(p=p), tf.bool),
                         body=lambda i: i + 1, loop_vars=[i])
  return sample


geom = geometric(p=0.9)

sess = tf.Session()
samples = [sess.run(geom) for _ in range(1000)]
plt.hist(samples, bins='auto')
plt.show()
