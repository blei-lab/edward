import edward as ed
import tensorflow as tf
#sg = tf.contrib.bayesflow.stochastic_graph
#sg = ed.models.stochastic_graph
sg = ed.models.stochastic_graph_reduced
distributions = tf.contrib.distributions

# here, we build the meta-graph for both the probability model and
# variational model; no graph is harmed in the process of their making
pg = tf.Graph()
with pg.as_default():
    # Define operations and tensors in `g`.
    # PROBABILITY MODEL
    mu = [0.0, 0.0, 0.0]
    sigma = tf.constant([1.1, 1.2, 1.3])
    pz = sg.DistributionTensor(distributions.Normal, mu=mu, sigma=sigma)
    # TODO this doesn't work explicitly, because this fixes samples
    pz_variable = tf.Variable(tf.identity(pz), trainable=False)
    #
    x = sg.DistributionTensor(distributions.Normal, mu=pz_variable, sigma=sigma)

qg = tf.Graph()
with qg.as_default():
    # VARIATIONAL MODEL
    mu2 = [1.0, 1.0, 1.0]
    sigma = tf.constant([1.1, 1.2, 1.3])
    qz = sg.DistributionTensor(distributions.Normal, mu=mu2, sigma=sigma)

# INFERENCE
# TODO
latent_vars = {z: qz}
data = {x: np.array([0.0, 0.0, 0.0])}

# here we use the meta-graphs to create the true graph
#init = tf.initialize_all_variables()
sess = tf.Session()
#sess.run(init)

# log p(x, z) for fixed data x and z ~ p(z)
p_log_prob = 0.0
# Sum over prior.
z_sample = pz.value()
p_log_prob += pz._dist.log_pdf(z_sample)
# Sum over likelihood.
p_log_prob += likelihood._dist.log_pdf(data)

sess.run(p_log_prob)

# log p(x, z) for fixed data x and z ~ q(z)
p_log_prob = 0.0
# Sum over prior.
z_sample = qz.value()
p_log_prob += pz._dist.log_pdf(z_sample)
# Sum over likelihood.
sess.run(pz_variable.assign(z_sample))
p_log_prob += likelihood._dist.log_pdf(data)

sess.run(p_log_prob)




###  TENSORFLOW VARIABLE META-GRAPH


# We define a static computational graph
#
# pbeta -> [ ] -> pz -> [ ] -> x
#
# where [] denotes variable. This allows us to define new tensors to
# throw into these slots, re-using the computational graph to then
# define
#
# qbeta -> [ ] -> qz -> [ ] -> x
#
# 1. Matt and my naive approach of halting the process, by throwing in
# samples x.log_prob(data, z=qz.sample()) doesn't work because there can be
# arbitrary tensor operations to form p(x | z); and we don't want to
# directly set the parameters of x | z; we want to replace the
# stochastic tensors in its dependence.
# 2. Ideally, we'd like some delayed graph construction where we can
# directly tell to do, e.g., qbeta -> qz -> x if the computation
# demands it. But knowing this structure to put into the delayed graph
# construction requires building such knowledge in the first place.
# Hence the tf.Variables().
# 3. THIS SOLUTION.
# It will re-assign all TensorFlow variables it depends on, from prior
# to variational if we are doing something like sampling. To do this,
# each distribution will have a dependency graph tracing all
# TensorFlow variables which it depends on and whose
# initialized_value() is a stochastictensor?
# The only weird thing is that re-assignment calls running the
# session; may be can form all assign ops as a long list, which is
# then run after construction of the graph, which will only initialize
# variables that are not these pseudo-ones.
# This is essentially the delayed graph methodology, but levering a
# graph construction to build the later delayed graph but moving
# around node dependencies.
#
# TODO the naive tf.Variable() leads to a fixed value for z
# therefore consider it as a "meta-graph" which can then be applied
# for building the real computational graphs

## EXPERIMENTATION WITH SHAPES

with sg.value_type(sg.SampleValue(n=5)):
  # (n, shape(pz))
  pz = sg.DistributionTensor(distributions.Normal, mu=0.0, sigma=1.0)
  # (n, shape), where shape is (n, shape(pz))
  likelihood = sg.DistributionTensor(distributions.Normal, mu=pz, sigma=1.0)

# mean across its own samples, (n, shape) -> (shape)
# [ mean p(x | z^1), ..., mean p(x | z^s) ]
sess.run(likelihood._dist.mean())

# mean p(x) = mean int p(x | z) p(z) dz
#           \approx 1/S \sum_{s=1}^S mean p(x | z^s)
sess.run(tf.reduce_mean(likelihood._dist.mean()))
# It can be optimized at compile time, in order to exactly calculate
# with marginal inference an analytic normal marginal density. This is
# key to Hakaru.

# [ log p(x | z^1), ..., log p(x | z^s) ]
sess.run(likelihood._dist.log_pdf(0.0))
# broadcasting error (2, ) and (1, 3)
sess.run(likelihood._dist.log_pdf([0.0, 1.0]))
#
# log p(x) = int log p(x | z) p(z) dz
#          \approx 1/S \sum_{s=1}^S log p(x | z^s)
sess.run(tf.reduce_mean(likelihood._dist.log_pdf(0.0)))


x = tf.Variable(0.0)
y = sg.DistributionTensor(distributions.Normal, mu=x, sigma=1.0)

y_log_pdf = y._dist.log_pdf(0.0)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
sess.run(y_log_pdf)
sess.run(y_log_pdf)

sess.run(x.assign(1.0))
y_log_pdf_new = y._dist.log_pdf(0.0)
sess.run(y_log_pdf_new)
sess.run(y_log_pdf_new)

#sess = tf.Session()
#sess.run(tf.identity(prior))
#sess.run(prior + tf.constant(5.0))
#sess.run([prior.mean(), prior.value()])

#import edward as ed
#import tensorflow as tf

##x = ed.RandomVariable()
#x = ed.Normal(loc=tf.constant(0.0), scale=tf.constant(1.0))
#x + tf.constant(5.0)

#sess = tf.Session()
#sess.run(x)
#sess.run(tf.identity(x))
#sess.run(x + tf.constant(5.0))
