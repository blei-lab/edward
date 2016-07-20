#!/usr/bin/env python
"""
Probabilistic matrix factorization.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational, Normal
from edward.stats import norm, poisson


class MatrixFactorization:
    """
    p(x, z) = [ prod_{i=1}^N prod_{j=1}^N Poi(Y_{ij}; \exp(s_iTt_j) ) ]
              [ prod_{i=1}^N N(s_i; 0, var) N(t_i; 0, var) ]

    where z = {s,t}.
    """
    def __init__(self, K, n_rows, n_cols=None, var=0.01,
                 like ='Poisson',
                 prior='Lognormal',
                 interaction ='additive'):
        if n_cols == None:
             n_cols = n_rows

        self.n_vars = (n_rows+n_cols)* K
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.K = K
        self.prior_variance = var
        self.like = like
        self.prior = prior
        self.interaction = interaction


    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        if self.prior == 'Lognormal':
            zs = tf.exp(zs)
        elif self.prior != 'Gaussian':
            raise NotImplementedError("prior not available.")

        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs)

        s = tf.reshape(zs[:,:self.n_rows*self.K], [self.n_rows,self.K])
        t = tf.reshape(zs[:,self.n_cols*self.K:], [self.n_cols,self.K])

        xp = tf.matmul(s, t, transpose_b=True)
        if self.interaction == 'multiplicative':
            xp = tf.exp(xp)
        elif self.interaction != 'additive':
            raise NotImplementedError("interaction type unknown.")

        if self.like == 'Gaussian':
            log_lik = tf.reduce_sum(norm.logpdf(xs['x'], xp))
        elif self.like == 'Poisson':
            if not (self.interaction == "additive" or self.prior == "Lognormal"):
                raise NotImplementedError("Rate of Poisson has to be nonnegatve.")

            log_lik = tf.reduce_sum(poisson.logpmf(xs['x'], xp))
        else:
            raise NotImplementedError("likelihood not available.")

        return log_lik + log_prior


def load_celegans_brain():
    x = np.load('data/celegans_brain.npy')
    N = x.shape[0]
    return {'x': x}, N


ed.set_seed(42)
data, N = load_celegans_brain()
K = 3
model = MatrixFactorization(K, N,
                            like='Poisson',
                            prior='Lognormal',
                            interaction='additive')

inference = ed.MAP(model, data)

#variational = Variational()
#variational.add(Normal(model.n_vars))
#inference = ed.MFVI(model, variational,data)

inference.run(n_iter=5000, n_print=500)
