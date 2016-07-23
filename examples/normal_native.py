#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Normal
Variational model
    Likelihood: Mean-field Normal
"""
import edward as ed
import tensorflow as tf

from edward.models import Normal

ed.set_seed(42)

mu = tf.constant([1.0])
std = tf.constant([1.0])
z = Normal(1, loc=mu, scale=std)

qz = Normal()

inference = ed.MFVI({z: qz})
inference.run(n_iter=10000)
