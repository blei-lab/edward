# Stan's Beta-Bernoulli model example
import numpy as np
import tensorflow as tf

import blackbox as bb
from blackbox.dists import bernoulli_log_prob, beta_log_prob
from blackbox.likelihoods import MFBeta

class BernoulliModel:
    """
    p(z) = Beta(z; 1, 1)
    p(x|z) = Bernoulli(x; z)
    """
    def __init__(self, data):
        self.data = data
        self.num_vars = 1

    def log_prob(self, zs):
        log_prior = beta_log_prob(zs[:, 0], alpha=1.0, beta=1.0)
        log_lik = tf.pack([
            tf.reduce_sum(bernoulli_log_prob(self.data, z)) \
            for z in tf.unpack(zs)])
        return log_lik + log_prior

np.random.seed(42)
tf.set_random_seed(42)

data = tf.constant((0,1,0,0,0,0,0,0,0,1), dtype=tf.float32)
model = BernoulliModel(data)
q = MFBeta(model.num_vars)

inference = bb.VI(model, q, 100)

loss = inference.build_loss()
update = tf.train.AdamOptimizer(0.1).minimize(-loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
feed_dict = {}
for t in range(1000):
    zs = inference.sample(sess)

    _, a, b, elbos = sess.run([update, inference.q.alpha, inference.q.beta, inference.elbo], {inference.zs: zs})

    if t % 100 == 0:
        print "iter %d alpha %.3f beta %.3f elbo %.2f " \
        % (t, a, b, np.mean(elbos))
