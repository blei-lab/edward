#!/usr/bin/env python
"""
Simple mixture of two gaussians
The variaitonal distribution is set to be one-dimentional gaussian
"""
import tensorflow as tf
import edward as ed
import numpy as np
from edward.stats import norm
from edward.util import get_dims
import matplotlib.pyplot as plt

class GaussMixture:
    def __init__(self, p1, mu1, Sigma1, mu2, Sigma2):
        self.num_vars = 1
        self.p1 = p1
        self.mu1 = mu1
        self.mu2 = mu2
        self.Sigma1 = Sigma1
        self.Sigma2 = Sigma2

    def log_prob(self, xs, zs):
        
        return tf.concat(0 ,  [tf.log(tf.exp(tf.log(self.p1) + norm.logpdf(z, self.mu1, self.Sigma1)) + 
             tf.exp(tf.log(1 - self.p1) + norm.logpdf(z, self.mu2, self.Sigma2)))
             for z in tf.unpack(zs)])

ed.set_seed(42)
mu1 = tf.constant(1.0)
mu2 = tf.constant(-2.0)
Sigma1 = tf.constant(1.0)
Sigma2 = tf.constant(2.0)

model = GaussMixture(0.8, mu1, Sigma1, mu2, Sigma2)
variational = ed.MFGaussian(model.num_vars)

#infer = ed.MFVI(model, variational)
infer = ed.ChiVI(model, variational) 
infer.run(n_iter=10000)

#### plots
from scipy.stats import norm
fig, ax = plt.subplots(1, 1)
x = np.linspace(-5,5, 1000)
true_val = 0.8 * norm.pdf(x, 1.0, 1.0) + 0.2 * norm.pdf(x, -2.0, 2.0)
chi_val = norm.pdf(x, -0.59536922, 0.84493601)
kl_val = norm.pdf(x, 0.77200621,  1.23507607)
ax.plot(x, true_val,'r-', lw=3, label='true distribution', color='red')
ax.plot(x, kl_val,'r-', lw=3, label='kl(q || p)', color='blue')
ax.plot(x, chi_val,'r-', lw=3, label='chi', color='green')
ax.legend(loc='best', frameon=False)                                              
ax.set_title("Mixture of Two Gaussians N(1, 1) and N(-2, 2)...n = 2.0...(0.8, 0.2)")
plt.show()

