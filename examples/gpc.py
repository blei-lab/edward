#!/usr/bin/env python
"""Gaussian process classification using mean-field variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, MultivariateNormalFull, Normal
from edward.util import multivariate_rbf_kernel

ed.set_seed(54)
# DATA
df = np.loadtxt('data/crabs_train.txt', dtype='float32', delimiter=',')
df[df[:, 0] == -1, 0] = 0
N = len(df)
D = df.shape[1] - 1
permutation = np.random.choice(range(N), N, replace=False)
X_train = df[:, 1:][permutation]
y_train = df[:, 0][permutation]

print("pre-computing the kernel matrix...")
K = multivariate_rbf_kernel(
    tf.convert_to_tensor(X_train), sigma=1.0, l=1.0)

# MODEL
X = ed.placeholder(tf.float32, [N, D])
f = MultivariateNormalFull(mu=tf.zeros(N), sigma=kernel(X))
y = Bernoulli(logits=f)

# INFERENCE
qf = Normal(mu=tf.Variable(tf.random_normal([N])),
            sigma=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))

data = {X: X_train, y: y_train}
inference = ed.KLqp({f: qf}, data)
inference.run(n_iter=500)
