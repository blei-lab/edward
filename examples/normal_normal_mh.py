#!/usr/bin/env python
"""Normal-normal model using Metropolis-Hastings."""
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
x_data = np.array([0.0] * 50)

# MODEL: Normal-Normal with known variance
mu = Normal(loc=0.0, scale=1.0)
x = Normal(loc=tf.ones(50) * mu, scale=1.0)

# INFERENCE
qmu = Empirical(params=tf.Variable(tf.zeros([1000])))

proposal_mu = Normal(loc=0.0, scale=tf.sqrt(1.0 / 51.0))

# analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
inference = ed.MetropolisHastings({mu: qmu}, {mu: proposal_mu},
                                  data={x: x_data})
inference.run()

# CRITICISM
# Check convergence with visual diagnostics.
sess = ed.get_session()
mean, stddev = sess.run([qmu.mean(), qmu.stddev()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior stddev:")
print(stddev)

# Check convergence with visual diagnostics.
samples = sess.run(qmu.params)

# Plot histogram.
plt.hist(samples, bins='auto')
plt.show()

# Trace plot.
plt.plot(samples)
plt.show()
