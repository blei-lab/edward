#!/usr/bin/env python
"""
A simple demonstration of how to implement new inference algorithms in
Edward. Here we implement importance-weighted variational inference.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.inferences import MFVI
from edward.models import Variational, Beta
from edward.stats import bernoulli, beta
from edward.util import log_mean_exp, stop_gradient


class IWVI(MFVI):
    """
    Importance-weighted variational inference. Uses importance
    sampling to produce an improved lower bound on the log marginal
    likelihood.

    It is the core idea behind importance-weighted autoencoders (Burda
    et al. (2016)). IWAEs are the special case when the probabilistic
    model is among a specific class of deep generative models, and the
    variational model is parameterized with an inference network.

    Notes
    -----
    `IWVI` is implemented by inheriting from mean-field (black box)
    variational inference (`MFVI`). The loss function to optimize is
    modified to include importance weights.
    """
    def __init__(self, *args, **kwargs):
        super(IWVI, self).__init__(*args, **kwargs)

    def initialize(self, K=5, *args, **kwargs):
        """Initialization.

        Parameters
        ----------
        K : int, optional
            Number of importance samples.
        """
        self.K = K
        return super(IWVI, self).initialize(*args, **kwargs)

    def build_loss(self):
        if self.score:
            return self.build_score_loss()
        else:
            return self.build_reparam_loss()

    def build_score_loss(self):
        """Build loss function. Its automatic differentiation
        is a stochastic gradient of

        .. math::

            -E_{q(z^1; \lambda), ..., q(z^K; \lambda)} [
            \log 1/K \sum_{k=1}^K p(x, z^k)/q(z^k; \lambda) ]

        based on the score function estimator. (Paisley et al., 2012)

        Computed by sampling from :math:`q(z;\lambda)` and evaluating
        the expectation using Monte Carlo sampling. Note there is a
        difference between the number of samples to approximate the
        expectations (`n_samples`) and the number of importance
        samples to determine how many expectations (`K`).
        """
        x = self.data
        losses = []
        for s in range(self.n_samples):
            z = self.variational.sample(self.K)
            p_log_prob = self.model.log_prob(x, z)
            q_log_prob = self.variational.log_prob(stop_gradient(z))
            log_w = p_log_prob - q_log_prob
            losses += [log_mean_exp(log_w)]

        losses = tf.pack(losses)
        self.loss = tf.reduce_mean(losses)
        return -tf.reduce_mean(q_log_prob * stop_gradient(losses))

    def build_reparam_loss(self):
        """Build loss function. Its automatic differentiation
        is a stochastic gradient of

        .. math::

            -E_{q(z^1; \lambda), ..., q(z^K; \lambda)} [
            \log 1/K \sum_{k=1}^K p(x, z^k)/q(z^k; \lambda) ]

        based on the reparameterization trick. (Kingma and Welling, 2014)

        Computed by sampling from :math:`q(z;\lambda)` and evaluating
        the expectation using Monte Carlo sampling. Note there is a
        difference between the number of samples to approximate the
        expectations (`n_samples`) and the number of importance
        samples to determine how many expectations (`K`).
        """
        x = self.data
        for s in range(self.n_samples):
            z = self.variational.sample(self.K)
            p_log_prob = self.model.log_prob(x, z)
            q_log_prob = self.variational.log_prob(z)
            log_w = p_log_prob - q_log_prob
            losses += [log_mean_exp(log_w)]

        losses = tf.pack(losses)
        self.loss = tf.reduce_mean(losses)
        return -self.loss


class BetaBernoulli:
    """p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)"""
    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs, a=1.0, b=1.0)
        log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs['x'], z))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior


ed.set_seed(42)
model = BetaBernoulli()
variational = Variational()
variational.add(Beta())
data = {'x': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

inference = IWVI(model, variational, data)
inference.run(K=10, n_iter=10000)
