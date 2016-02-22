# Part of code used in _py_log_prob() in StanModel class.
import numpy as np
import tensorflow as tf
import pystan

from collections import OrderedDict

schools_code = """
data {
    int<lower=0> J; // number of schools
    real y[J]; // estimated treatment effects
    real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
    real mu;
    real<lower=0> tau;
    real eta[J];
}
transformed parameters {
    real theta[J];
    for (j in 1:J)
    theta[j] <- mu + tau * eta[j];
}
model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
}
"""

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

model = pystan.stan(model_code=schools_code, data=schools_dat,
                    iter=1000, chains=4)
temp = model.extract()
temp.pop(u'lp__')

# Original z, a sample of dictionary type extracted from the model.
z_orig = OrderedDict()
for i, par in enumerate(model.model_pars):
    z_orig[par] = temp[par][0]

# Flattened version, what we get during inference as a flattened array.
temp = z_orig.values()
z_orig_flat = []
for elem in temp:
    if isinstance(elem, np.ndarray):
        z_orig_flat += list(elem)
    else:
        z_orig_flat += [elem]

z_orig_flat = np.array(z_orig_flat)

# To use unconstrain_pars(), convert it into the dictionary type.
z = OrderedDict()
idx = 0
for dim, par in zip(model.par_dims, model.model_pars):
    elems = np.sum(dim)
    if elems == 0:
        z[par] = float(z_orig_flat[idx])
        idx += 1
    else:
        z[par] = z_orig_flat[idx:(idx+elems)].reshape(dim)
        idx += elems

model.unconstrain_pars(z)
