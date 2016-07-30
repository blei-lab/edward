from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.util import get_session

#sg = tf.contrib.bayesflow.stochastic_graph
import edward as ed
sg = ed.models.stochastic_graph


class VariationalInference(object):
    def __init__(self, latent_vars, data):
        # TODO prevent build for data which are already tensor's (e.g.
        # placeholder in the setting above)
        # what are other cases? think about them all according to modeling
        # spec and thus the inputs/outputs of latent_vars and data
        self.latent_vars = latent_vars
        self.data = data

    def initialize(self, n_samples=1):
        # Use a dictionary to store bindings of `RandomVariable`s to their built tensors.
        self.built_dict = {}
        with sg.value_type(sg.SampleValue(n=n_samples)): # latent variable samples
            # Build random variables in q(z).
            for rv in six.itervalues(self.latent_vars):
                rv.build(built_dict=self.built_dict)

            # Build random variables in p(z). `latent_vars` replaces conditioning on
            # priors with conditioning on (variational) posteriors.
            for rv in six.iterkeys(self.latent_vars):
                rv.build(built_dict=self.built_dict, latent_vars=self.latent_vars)

        with sg.value_type(sg.SampleAndReshapeValue(n=1)): # 1 data set
            # Build random variables in p(x | z). `latent_vars` replaces conditioning on
            # priors with conditioning on (variational) posteriors.
            for rv in six.iterkeys(self.data):
                # TODO don't call rv
                if not isinstance(rv, tf.Tensor):
                    # TODO for placeholders as in covariates or pre-built
                    # model tensors
                    rv.build(built_dict=self.built_dict, latent_vars=self.latent_vars)

        self.loss = self.build_loss()
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)

    def build_loss(self):
        p_log_prob = 0.0
        q_log_prob = 0.0
        # Take log-densities over latent variables.
        for pz, qz in six.iteritems(self.latent_vars):
            pz_tensor = self.built_dict[pz]
            qz_tensor = self.built_dict[qz]
            z_samples = qz_tensor.value() # (n_samples, shape) tensor
            # Sum over all dimensions except the one corresponding to n_samples.
            q_log_prob += tf.reduce_sum(qz_tensor.distribution.log_pdf(z_samples),
                                        range(1, len(qz_tensor.value().get_shape())))
            p_log_prob += tf.reduce_sum(pz_tensor.distribution.log_pdf(z_samples),
                                        range(1, len(pz_tensor.value().get_shape())))

        # Take log-densities over data.
        for px, obs in six.iteritems(self.data):
            if not isinstance(px, tf.Tensor):
                px_tensor = self.built_dict[px]
                # reshape in order to broadcast along outer dimension
                obs = tf.reshape(obs, obs.shape + (1,)*(len(px_tensor.value().get_shape())-1))
                # Sum over all dimensions except the one corresponding to n_samples.
                p_log_prob += tf.reduce_sum(px_tensor.distribution.log_pdf(obs),
                                            [0] + range(2, len(px_tensor.value().get_shape())))

        # -ELBO, whose automatic differentiation via
        # reparameterization trick is a stochastic gradient.
        return -tf.reduce_mean(p_log_prob - q_log_prob)
