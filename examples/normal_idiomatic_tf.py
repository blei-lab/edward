#!/usr/bin/env python
"""This demonstrates a more idiomatic TensorFlow example, which
provides more fine experimentation. We do not call inference.run().
Alternatively, we directly manipulate various objects during
inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal

ed.set_seed(42)

# MODEL
z = Normal(loc=1.0, scale=1.0)

# INFERENCE
qz = Normal(loc=tf.Variable(tf.random_normal([])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

inference = ed.KLqp({z: qz})
inference.initialize(n_iter=250)

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
