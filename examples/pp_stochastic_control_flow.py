#!/usr/bin/env python
"""Stochastic control flow.

We sample from a geometric random variable by using samples from
Bernoulli random variable. It requires a while loop whose condition is
stochastic.

References
----------
https://probmods.org/generative-models.html#stochastic-recursion
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf

from edward.models import Bernoulli


def geometric(p):
    i = tf.constant(0)

    def cond(i):
      return tf.equal(Bernoulli(p=p), tf.constant(1))

    def body(i):
      return i + 1

    return tf.while_loop(cond, body, loop_vars=[i])


p = tf.constant(0.9)
geom = geometric(p)

sess = tf.Session()
samples = []
for n in range(1000):
    samples.append(sess.run(geom))

plt.hist(samples, bins='auto')
