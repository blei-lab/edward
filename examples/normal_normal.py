from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import Normal

#sg = tf.contrib.bayesflow.stochastic_graph
import edward as ed
sg = ed.models.stochastic_graph

# PROBABILITY MODEL
mu = tf.constant([0.0])
sigma = tf.constant([1.0])
pmu = Normal([mu, sigma])
# normal likelihood for 5 data points, with shared unobserved mean
x = Normal([pmu, sigma],
           lambda cond_set: tf.pack([cond_set[0] for n in range(5)]))

# VARIATIONAL MODEL
mu2 = tf.Variable(tf.random_normal([1]))
sigma2 = tf.nn.softplus(tf.Variable(tf.random_normal([1])))
qmu = Normal([mu2, sigma2])

data = {x: np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}

## GENERAL APPROACH

from edward.util import get_session

class Inference(object):
    def __init__(self, latent_vars, data):
        get_session()
        self.latent_vars = latent_vars
        self.data = data

class VariationalInference(Inference):
    def __init__(self, latent_vars, data):
        super(VariationalInference, self).__init__(latent_vars, data)

    def initialize(self):
        # Use a dictionary to store bindings of `RandomVariable`s to their built tensors.
        self.built_dict = {}
        n_samples = 4
        with sg.value_type(sg.SampleValue(n=n_samples)): # 4 latent variable samples
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
                rv.build(built_dict=self.built_dict, latent_vars=self.latent_vars)

        self.loss = self.build_loss()
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)

    def build_loss(self):
        p_log_prob = 0.0
        q_log_prob = 0.0
        # Sum over prior.
        for pz, qz in six.iteritems(self.latent_vars):
            pz_tensor = self.built_dict[pz]
            qz_tensor = self.built_dict[qz]
            z_samples = qz_tensor.value() # (n_samples, shape) tensor
            # Sum over all dimensions except the one corresponding to n_samples.
            q_log_prob += tf.reduce_sum(qz_tensor.distribution.log_pdf(z_samples),
                                        range(1, len(qz_tensor.value().get_shape())))
            p_log_prob += tf.reduce_sum(pz_tensor.distribution.log_pdf(z_samples),
                                        range(1, len(pz_tensor.value().get_shape())))

        # Sum over likelihood.
        for px, obs in six.iteritems(self.data):
            px_tensor = self.built_dict[px]
            # reshape in order to broadcast along outer dimension
            obs = tf.reshape(obs, obs.shape + (1,)*(len(px_tensor.value().get_shape())-1))
            # Sum over all dimensions except the one corresponding to n_samples.
            p_log_prob += tf.reduce_sum(px_tensor.distribution.log_pdf(obs),
                                        [0] + range(2, len(px_tensor.value().get_shape())))

        # -ELBO, whose automatic differentiation via
        # reparameterization trick is a stochastic gradient.
        return -tf.reduce_mean(p_log_prob - q_log_prob)

inference = VariationalInference({pmu: qmu}, data)
inference.initialize()

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
# TODO debug; doesn't seem to converge correctly
for t in range(1000):
    _, loss = sess.run([inference.train, inference.loss])
    if t % 100 == 0:
        print("loss: {:0.3f}".format(loss))
        print(sess.run([mu2, sigma2]))
