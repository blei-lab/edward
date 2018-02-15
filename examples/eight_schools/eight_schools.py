"""Implement the stan 8 schools example using the recommended non-centred
parameterization.

The Stan example is slightly modified to avoid improper priors and
avoid half-Cauchy priors.  Inference is with Edward using both HMC
and KLQP.

This model has a hierachy and an inferred variance - yet the example is
very simple - only the Normal distribution is used.

#### References
https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
http://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
import numpy as np
from edward.models import Normal, Empirical


def main(_):
    # data
    J = 8
    data_y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    data_sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

    # model definition
    mu = Normal([0.], [1.])
    logtau = Normal([5.], [1.])
    theta_tilde = Normal(tf.zeros(J), tf.ones(J))
    sigma = tf.placeholder(tf.float32, J)
    y = Normal(mu + tf.exp(logtau) * theta_tilde, sigma * tf.ones([J]))

    data = {y: data_y, sigma: data_sigma}

    # ed.KLqp inference
    with tf.variable_scope('q_logtau'):
        q_logtau = Normal(tf.get_variable('loc', [1]),
                          tf.nn.softplus(tf.get_variable('scale', [1])))

    with tf.variable_scope('q_mu'):
        q_mu = Normal(tf.get_variable('loc', [1]),
                      tf.nn.softplus(tf.get_variable('scale', [1])))

    with tf.variable_scope('q_theta_tilde'):
        q_theta_tilde = Normal(tf.get_variable('loc', [J]),
                               tf.nn.softplus(tf.get_variable('scale', [J])))

    inference = ed.KLqp({logtau: q_logtau, mu: q_mu,
                        theta_tilde: q_theta_tilde}, data=data)
    inference.run(n_samples=15, n_iter=60000)
    print("====    ed.KLqp inference ====")
    print("E[mu] = %f" % (q_mu.mean().eval()))
    print("E[logtau] = %f" % (q_logtau.mean().eval()))
    print("E[theta_tilde]=")
    print((q_theta_tilde.mean().eval()))
    print("====  end ed.KLqp inference ====")
    print("")
    print("")

    # HMC inference
    S = 400000
    burn = S // 2
    hq_logtau = Empirical(tf.Variable(tf.zeros([S, 1])))
    hq_mu = Empirical(tf.Variable(tf.zeros([S, 1])))
    hq_theta_tilde = Empirical(tf.Variable(tf.zeros([S, J])))

    inference = ed.HMC({logtau: hq_logtau, mu: hq_mu,
                        theta_tilde: hq_theta_tilde}, data=data)
    inference.run()

    print("====    ed.HMC inference ====")
    print("E[mu] = %f" % (hq_mu.params.eval()[burn:].mean()))
    print("E[logtau] = %f" % (hq_logtau.params.eval()[burn:].mean()))
    print("E[theta_tilde]=")
    print(hq_theta_tilde.params.eval()[burn:, ].mean(0))
    print("====  end ed.HMC inference ====")
    print("")
    print("")


if __name__ == "__main__":
  tf.app.run()
