#!/usr/bin/env python                                                                                                            

"""                                                                                                                     
Simple gamma model                                                                                                   
The variaitonal distribution is set to be a gamma distribution                                                                
"""
import tensorflow as tf
import edward as ed
import numpy as np
from edward.stats import gamma
from edward.util import get_dims, log_gamma
import matplotlib.pyplot as plt


class Gamma_Model:
     def __init__(self, a, b):
        self.num_vars = 1
        self.a = a
        self.b = b

     def log_prob(self, xs, zs):

         return tf.concat(0, [(self.a - 1.0) * tf.log(z) - \
                               z/self.b - self.a * tf.log(self.b) - log_gamma(self.a)
                               for z in tf.unpack(zs)])


ed.set_seed(42)
a = tf.constant(2.0)
b = tf.constant(2.0)
model = Gamma_Model(a, b)
variational = ed.MFGamma(model.num_vars)
infer_kl = ed.MFVI(model, variational)                                                                           
infer_chi = ed.ChiVI(model, variational)
infer_kl.run(n_iter=10000)
infer_chi.run(n_iter=10000)

####plots
from scipy.stats import gamma
fig, ax = plt.subplots(1, 1)
x = np.linspace(0,20, 1000)
true_val = gamma.pdf(x, a=2.0, scale=2.0)
#print(true_val)
chi_val = gamma.pdf(x, a=2.03312111, scale=1.98013365)
kl_val = gamma.pdf(x,  a=1.99984527, scale=2.00015569)

ax.plot(x, true_val,'r', lw=3, label='true distribution', color='red')
ax.plot(x, kl_val,'r-', lw=3, label='kl(q || p)', color='blue')
ax.plot(x, chi_val,'r-', lw=3, label='chi', color='green')
ax.legend(loc='best', frameon=False)
ax.set_title("Gamma(1, 2) Model...n = 1.1")
plt.show()



