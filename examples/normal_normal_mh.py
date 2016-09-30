#!/usr/bin/env python
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
qmu = Empirical(params=tf.Variable(tf.zeros([1000])))

proposal_mu = Normal(mu=0.0, sigma=tf.sqrt(1.0 / 51.0))

# analytic solution: N(mu=0.0, sigma=\sqrt{1/51}=0.140)
data = {x: x_data}
inference = ed.MetropolisHastings({mu: qmu}, {mu: proposal_mu}, data)
inference.run()

# Check convergence with visual diagnostics.
sess = ed.get_session()
samples = sess.run(qmu.params)

# Plot histogram.
plt.hist(samples, bins='auto')
plt.show()

# Trace plot.
plt.plot(samples)
plt.show()
