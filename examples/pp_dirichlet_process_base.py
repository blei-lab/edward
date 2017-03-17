#!/usr/bin/env python
"""Dirichlet process.

We sample from a Dirichlet process (with inputted base distribution)
via its stick breaking construction. For details on the
implementation, see the source for ``DirichletProcess``.

References
----------
https://probmods.org/chapters/12-non-parametric-models.html#infinite-discrete-distributions-the-dirichlet-processes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models import DirichletProcess, Exponential, Normal

base_cls = Normal
kwargs = {'mu': 0.0, 'sigma': 1.0}
dp = DirichletProcess(0.1, base_cls, **kwargs)
print(dp)

# ``theta`` is the distribution indirectly returned by the DP.
theta = base_cls(value=tf.cast(dp, tf.float32), **kwargs)
print(theta)

# Fetching theta is the same as fetching the Dirichlet process.
sess = tf.Session()
print(sess.run([dp, theta]))
print(sess.run([dp, theta]))

# This also works for non-scalar concentrations and base distributions.
base_cls = Exponential
kwargs = {'lam': tf.ones([5, 2])}
dp = DirichletProcess(tf.constant([0.1, 0.6, 0.4]), base_cls, **kwargs)
print(dp)
