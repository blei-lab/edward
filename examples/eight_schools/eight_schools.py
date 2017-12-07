# implement the stan 8 schools example
# using the recommended non-centred parameterisation
#
# https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
# http://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html
# we slightly modify the stan example to:
#                                         avoid improper priors
#                                         avoid half-cauchy priors
#                                         avoid informative priors
# make the statement of the model directly comparable
#
# This model has a hierachy and an inferred variance - yet the example is
# very simple - only the Normal distribution is used

import edward as ed
import tensorflow as tf
from edward.models import Normal, Empirical
import numpy as np

# data
J = 8
data_y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
data_sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
# end data

# model definition
mu = Normal(0.*tf.ones([1]), 10.*tf.ones([1]))
logtau = Normal(0.*tf.ones([1]), 1.*tf.ones([1]))
theta_tilde = Normal(tf.zeros([J]), tf.ones([J]))
sigma = tf.placeholder(tf.float32, J)
y = Normal(mu + tf.exp(logtau) * theta_tilde, sigma*tf.ones(J))
# end model definition

thedata = {y: data_y, sigma: data_sigma}

# ed.KLqp inference
q_logtau = Normal(tf.Variable(tf.random_normal([1])), tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
q_mu = Normal(tf.Variable(tf.random_normal([1])), tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
q_theta_tilde = Normal(tf.Variable(tf.random_normal([J])), tf.nn.softplus(tf.Variable(tf.random_normal([J]))))
inference = ed.KLqp({logtau: q_logtau, mu: q_mu, theta_tilde: q_theta_tilde}, data=thedata)
inference.run(n_samples=15, n_iter=20000)
# end ed.KLqp inference
print "====    ed.KLqp inference ===="
print "E[mu] = %f" % (q_mu.mean().eval())
print "E[logtau] = %f" % (q_logtau.mean().eval())
print "E[theta_tilde]="
print (q_theta_tilde.mean().eval())
print "====  end ed.KLqp inference ===="
print ""
print ""

# HMC inference
S = 100000
hq_logtau = Empirical(tf.Variable(tf.zeros([S, 1])))
hq_mu = Empirical(tf.Variable(tf.zeros([S, 1])))
hq_theta_tilde = Empirical(tf.Variable(tf.zeros([S, J])))

inference = ed.HMC({logtau: hq_logtau, mu: hq_mu, theta_tilde: hq_theta_tilde}, data=thedata)
inference.run()
# end HMC inference

print "====    ed.HMC inference ===="
print "E[mu] = %f" % (hq_mu.mean().eval())
print "E[logtau] = %f" % (hq_logtau.mean().eval())
print "E[theta_tilde]="
print (hq_theta_tilde.mean().eval())
print "====  end ed.HMC inference ===="
print ""
print ""

print "In order to see the result using Stan/NUTS run:"
print "Rscript eight_schools.R"


try:
    import pystan
    standata = dict(J=J, y=data_y, sigma=data_sigma)
    fit = pystan.stan('eight_schools.stan', data=standata, iter=100000)
    print(fit)
except ImportError:
    print "pystan not detected"
    print "if you have Rstan try: Rscript eight_schools.R"
