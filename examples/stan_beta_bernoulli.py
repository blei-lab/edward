#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.

The model is written in Stan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Beta

model_code = """
  data {
    int<lower=0> N;
    int<lower=0,upper=1> x[N];
  }
  parameters {
    real<lower=0,upper=1> p;
  }
  model {
    p ~ beta(1.0, 1.0);
    for (n in 1:N)
    x[n] ~ bernoulli(p);
  }
"""
ed.set_seed(42)
data = {'N': 10, 'x': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]}

model = ed.StanModel(model_code=model_code)

qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp = Beta(a=qp_a, b=qp_b)

inference = ed.KLqp({'p': qp}, data, model)
inference.run(n_iter=500)
