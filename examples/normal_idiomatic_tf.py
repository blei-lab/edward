#!/usr/bin/env python
"""
This demonstrates a more idiomatic TensorFlow example. Instead of
running inference.run(), we may want direct access to the TensorFlow
session and to manipulate various objects during inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal

ed.set_seed(42)

z = Normal(mu=1.0, sigma=1.0)

qz = Normal(mu=tf.Variable(tf.random_normal([])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

inference = ed.MFVI({z: qz})
inference.initialize(n_print=50)

init = tf.initialize_all_variables()
init.run()

for _ in range(250):
  info_dict = inference.update()
  inference.print_progress(info_dict)
