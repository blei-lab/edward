import numpy as np
import tensorflow as tf

class VI:
    def __init__(self, model, q, n_iter=1000, n_minibatch=1):
        self.model = model
        self.q = q

        self.n_iter = n_iter
        self.n_minibatch = n_minibatch

        self.zs = tf.placeholder(shape=(self.n_minibatch, q.num_vars),
                                 dtype=tf.float32,
                                 name="zs")
        self.elbo = 0

    def sample(self, sess):
        return self.q.sample(self.zs.get_shape(), sess)

    def build_score_loss(self):
        # TODO use MFVI gradient
        q_log_prob = tf.zeros([self.n_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, self.zs)

        self.elbo = self.model.log_prob(self.zs) - q_log_prob
        return tf.reduce_mean(q_log_prob * tf.stop_gradient(self.elbo))

    def run(self):
        sess, update = self._initialize()
        for t in range(self.n_iter):
            self._update(t, sess, update)

    def _initialize(self):
        loss = self.build_score_loss()
        update = tf.train.AdamOptimizer(0.1).minimize(-loss)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        return sess, update

    def _update(self, t, sess, update):
        zs = self.sample(sess)
        _, elbos = sess.run([update, self.elbo], {self.zs: zs})

        if t % 100 == 0:
            print "iter %d elbo %.2f " % (t, np.mean(elbos))
            self.q.print_params(sess)

# TODO keep porting stuff
# TODO be consistent with tf.dtypes
# TODO visualize graph
