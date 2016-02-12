# Bernoulli posterior, no data
import numpy as np
import tensorflow as tf

import blackbox as bb
from blackbox.dists import bernoulli_log_prob
from blackbox.likelihoods import MFBernoulli

class Bernoulli:
    """
    p(x, z) = p(z) = p(z | x) = Bernoulli(z; p)
    """
    def __init__(self, p):
        self.p = p
        self.num_vars = len(p.shape)

    def log_prob(self, zs):
        log_prior = bernoulli_log_prob(zs[:, 0], p)
        return log_prior

np.random.seed(42)
tf.set_random_seed(42)

p = np.array([0.6])
model = Bernoulli(p)
q = MFBernoulli(model.num_vars)

inference = bb.VI(model, q, 100)

loss = inference.build_loss()
update = tf.train.AdamOptimizer(0.1).minimize(-loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
feed_dict = {}
for t in range(1000):
    zs = inference.sample(sess)

    _, lamda, elbos = sess.run([update, inference.q.lamda, inference.elbo], {inference.zs: zs})

    if t % 100 == 0:
        print "iter %d p %.3f elbo %.2f " \
        % (t, 1.0 / (1.0 + np.exp(-lamda)), np.mean(elbos))
