#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal

sg = tf.contrib.bayesflow.stochastic_graph

ed.set_seed(42)

# Normal-Normal with known variance
mu = Normal(mu=tf.constant([0.0]), sigma=tf.constant([1.0]))
with sg.value_type(sg.SampleValue(n=50)):
    x = Normal(mu=mu, sigma=tf.constant([1.0]))

qmu_mu = tf.Variable(tf.random_normal([1]))
qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([1])))
qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)

data = {x: np.array([0.0]*50, dtype=np.float32)}

# analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
inference = ed.MFVI({mu: qmu}, data)
inference.initialize()

sess = ed.get_session()
for t in range(1001):
    _, loss = sess.run([inference.train, inference.loss])
    inference.print_progress(t, loss)
