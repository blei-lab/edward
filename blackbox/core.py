import numpy as np
import tensorflow as tf

from scipy.stats import norm

class VI:
    def __init__(self, model, q, method="score", n_iter=1000, n_minibatch=1):
        self.model = model
        self.q = q

        self.method = "score"
        self.n_iter = n_iter
        self.n_minibatch = n_minibatch

        self.samples = tf.placeholder(shape=(self.n_minibatch, q.num_vars),
                                 dtype=tf.float32,
                                 name="zs")
        self.elbo = 0

    def run(self):
        sess, update = self._initialize()
        for t in range(self.n_iter):
            self._update(t, sess, update)

    def _initialize(self):
        if self.method == "score":
            loss = self.build_score_loss()
        else:
            loss = self.build_reparam_loss()

        update = tf.train.AdamOptimizer(0.1).minimize(-loss)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        return sess, update

    def _update(self, t, sess, update):
        if self.method == "score":
            samples = self.q.sample(self.samples.get_shape(), sess)
        else:
            # TODO generalize to "noise" samples, and
            # reparameterization method, based on q's methods
            # TODO I could use tf.random_normal() here, although I
            # need it to realize values.
            samples = norm.rvs(size=self.samples.get_shape())

        _, elbos = sess.run([update, self.elbo], {self.samples: samples})

        if t % 100 == 0:
            print "iter %d elbo %.2f " % (t, np.mean(elbos))
            self.q.print_params(sess)

    def build_score_loss(self):
        # TODO use MFVI gradient
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, self.samples)

        self.elbo = self.model.log_prob(self.samples) - q_log_prob
        return tf.reduce_mean(q_log_prob * tf.stop_gradient(self.elbo))

    def build_reparam_loss(self):
        # TODO use MFVI gradient
        m = self.q.transform_m(self.q.m_unconst)
        s = self.q.transform_s(self.q.s_unconst)
        z = m + self.samples * s

        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, z)

        self.elbo = self.model.log_prob(z) - q_log_prob
        # TODO in the special case of gaussian entropy, calculate it
        # analytically
        return tf.reduce_mean(self.elbo)

# TODO keep porting stuff
# TODO be consistent with tf.dtypes
# TODO visualize graph
