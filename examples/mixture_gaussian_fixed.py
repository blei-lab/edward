#!/usr/bin/env python
"""
Two-component mixture of Gaussians
The variational distribution is set to be mean-field Gaussian
"""
import edward as ed
import tensorflow as tf
import numpy as np

from edward.stats import norm
from edward.util import get_dims
import matplotlib.pyplot as plt
from scipy import stats

class GaussMixture:
    """Two-component Gaussian mixture model."""
    def __init__(self, p1, mu1, Sigma1, mu2, Sigma2):
        self.num_vars = 1
        self.p1 = p1
        self.mu1 = mu1
        self.mu2 = mu2
        self.Sigma1 = Sigma1
        self.Sigma2 = Sigma2

    def log_prob(self, xs, zs):
        return tf.pack([tf.log( \
            tf.exp(tf.log(self.p1) + \
                   norm.logpdf(z, self.mu1, self.Sigma1)) + \
            tf.exp(tf.log(1 - self.p1) + \
                   norm.logpdf(z, self.mu2, self.Sigma2)))
            for z in tf.unpack(zs)])

ed.set_seed(42)
mu1 = tf.constant(1.0)
mu2 = tf.constant(-2.0)
Sigma1 = tf.constant(1.0)
Sigma2 = tf.constant(2.0)

model = GaussMixture(0.8, mu1, Sigma1, mu2, Sigma2)
variational = ed.MFGaussian(model.num_vars)

inference = ed.MFVI(model, variational)
inference.run(n_iter=1000)

fig, ax = plt.subplots(1, 1)
x = np.linspace(-5,5, 1000)
true_val = 0.8 * stats.norm.pdf(x, 1.0, 1.0) + 0.2 * stats.norm.pdf(x, -2.0, 2.0)
chi_val = stats.norm.pdf(x, -0.59536922, 0.84493601)
kl_val = stats.norm.pdf(x, 0.77200621,  1.23507607)
ax.plot(x, true_val,'r-', lw=3, label='true distribution', color='red')
ax.plot(x, kl_val,'r-', lw=3, label='kl(q || p)', color='blue')
ax.legend(loc='best', frameon=False)
ax.set_title("Mixture of Two Gaussians N(1, 1) and N(-2, 2)...n = 2.0...(0.8, 0.2)")
plt.show()
