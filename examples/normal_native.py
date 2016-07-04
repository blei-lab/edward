#!/usr/bin/env python
"""
Probability model
    Posterior: (1-dimensional) Normal
Variational model
    Likelihood: Mean-field Normal
"""
import edward as ed
import tensorflow as tf

from edward.models import Model, Normal

ed.set_seed(42)

mu = tf.constant([1.0])
std = tf.constant([1.0])
z = Normal(1, loc=mu, scale=std)

model = Model()
model.add(z)

qz = Normal()

variational = Model()
variational.add(qz)

# TODO hard-code the mapping
ed.MFVI.mapping = {qz: z}
inference = ed.MFVI(model, variational, data={})
inference.run(n_iter=10000)
