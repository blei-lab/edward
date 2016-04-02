#!/usr/bin/env python                                                                   
"""                                                                                     
Simple truncated gaussian model                                                         
The variaitonal distribution is set to be one-dimentional gaussian                      
"""
import tensorflow as tf
import edward as ed
import numpy as np
from edward.stats import truncatednorm
import matplotlib.pyplot as plt

class TruncGauss:
    def __init__(self, a, b, loc, scale):
        self.num_vars = 1
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        
    def log_prob(self, xs, zs):
        
        return tf.concat(0, [truncatednorm.logpdf(z, self.a, self.b, self.loc, self.scale)  for z in tf.unpack(zs)])


ed.set_seed(42)
a = tf.constant(-2.0)
b = tf.constant(6.0)
loc = tf.constant(0.0)
scale = tf.constant(1.0)
model = TruncGauss(a, b, loc, scale)
variational = ed.MFGaussian(model.num_vars)

infer_kl = ed.MFVI(model, variational)                                                    
infer_chi = ed.ChiVI(model, variational)
infer_kl.run(n_iter=10000)
infer_chi.run(n_iter=10000)

#plots
from scipy.stats import truncnorm, norm
fig, ax = plt.subplots(1, 1)
x = np.linspace(-6,6, 1000)
true_val = truncnorm.pdf(x, -2.0, 6.0, 0.0, 1.0) 
chi_val = norm.pdf(x, -0.02856113, 0.67771775)
kl_val = norm.pdf(x, 0.033 , 0.99480462)
ax.plot(x, true_val,'r-', lw=3, label='true distribution', color='red')
ax.plot(x, kl_val,'r-', lw=3, label='kl(q || p)', color='blue')
ax.plot(x, chi_val,'r-', lw=3, label='chi', color='green')
ax.legend(loc='best', frameon=False)
ax.set_title("Truncated Normal(0, 1) to (-2, 6)")
plt.show()



