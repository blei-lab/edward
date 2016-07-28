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
mu2 = tf.constant([1.0])
sigma2 = tf.constant([1.0])
qmu = Normal([mu2, sigma2])

# INFERENCE
latent_vars = {pmu: qmu}
data = {x: np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
# see previous gist; this is all that's necessary at inference time, i.e.,
# inference = VariationalInference(latent_vars, data)


## MANUAL APPROACH


# Build the stochastic tensors necessary for this joint density.
n_samples = 4
with sg.value_type(sg.SampleValue(n=n_samples)): # 4 latent variable samples
  qmu_tensor = qmu.build()
  pmu_tensor = pmu.build()

with sg.value_type(sg.SampleAndReshapeValue(n=1)): # 1 data set
  x_tensor = x.build([qmu_tensor, None]) # <- it depends on qmu_tensor!

# Calculate the joint density using these tensors.
# It is a vector
# [ log p(x, z^{1}), ..., log p(x, z^{n_samples}) ],
# summming over all dimensions except the one corresponding to n_samples.
p_log_prob_manual = 0.0
z_samples = qmu_tensor.value()
p_log_prob_manual += tf.reduce_sum(pmu_tensor.distribution.log_pdf(z_samples), 1)
obs = tf.reshape(data[x], [5, 1, 1]) # reshape in order to broadcast along outer dimension
p_log_prob_manual += tf.reduce_sum(x_tensor.distribution.log_pdf(obs), [0,2])

sess = tf.Session()
sess.run(p_log_prob_manual)


## GENERAL APPROACH


# Use a dictionary to store bindings of `RandomVariable`s to their built tensors.
built_dict = {}
n_samples = 4
with sg.value_type(sg.SampleValue(n=n_samples)): # 4 latent variable samples
    # Build random variables in q(z).
    for rv in six.itervalues(latent_vars):
        rv.build(built_dict=built_dict)

    # Build random variables in p(z). `latent_vars` replaces conditioning on
    # priors with conditioning on (variational) posteriors.
    for rv in six.iterkeys(latent_vars):
        rv.build(built_dict=built_dict, latent_vars=latent_vars)

with sg.value_type(sg.SampleAndReshapeValue(n=1)): # 1 data set
    # Build random variables in p(x | z). `latent_vars` replaces conditioning on
    # priors with conditioning on (variational) posteriors.
    for rv in six.iterkeys(data):
        rv.build(built_dict=built_dict, latent_vars=latent_vars)

p_log_prob = 0.0
# Sum over prior.
for pz, qz in six.iteritems(latent_vars):
    pz_tensor = built_dict[pz]
    qz_tensor = built_dict[qz]
    z_samples = qz_tensor.value() # (n_samples, shape) tensor
    # Sum over all dimensions except the one corresponding to n_samples.
    p_log_prob += tf.reduce_sum(pz_tensor.distribution.log_pdf(z_samples),
                                range(1, len(pz_tensor.value().get_shape())))

# Sum over likelihood.
for px, obs in six.iteritems(data):
    px_tensor = built_dict[px]
    # reshape in order to broadcast along outer dimension
    obs = tf.reshape(obs, obs.shape + (1,)*(len(x_tensor.value().get_shape())-1))
    # Sum over all dimensions except the one corresponding to n_samples.
    p_log_prob += tf.reduce_sum(px_tensor.distribution.log_pdf(obs),
                                [0] + range(2, len(px_tensor.value().get_shape())))

sess.run(p_log_prob)
