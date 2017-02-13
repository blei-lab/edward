#!/usr/bin/env python
"""Normal-normal model using Hamiltonian Monte Carlo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Empirical, Normal

ed.set_seed(42)

# DATA
x_data = np.array([0.0] * 50, dtype=np.float32)

# MODEL: Normal-Normal with known variance
mu = Normal(mu=0.0, sigma=1.0)
x = Normal(mu=tf.ones(50) * mu, sigma=1.0)

# INFERENCE
qmu = Empirical(params=tf.Variable(tf.zeros(1000)))

# analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
inference = ed.HMC({mu: qmu}, data={x: x_data})
inference.run()

# CRITICISM
sess = ed.get_session()
mean, std = sess.run([qmu.mean(), qmu.std()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior std:")
print(std)

# Check convergence with visual diagnostics.
samples = sess.run(qmu.params)

# Plot histogram.
plt.hist(samples, bins='auto')
plt.show()

# Trace plot.
plt.plot(samples)
plt.show()
