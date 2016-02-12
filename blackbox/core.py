import tensorflow as tf

class VI:
    def __init__(self, model, q, num_minibatch=1):
        self.num_minibatch = num_minibatch
        self.zs = tf.placeholder(shape=(self.num_minibatch, q.num_vars),
                                 dtype=tf.float32,
                                 name="zs")
        self.model = model
        self.q = q
        self.elbo = 0

    def sample(self, a, b):
        # TODO the size should be tf.shape(self.zs)
        return self.q.sample((self.num_minibatch, self.q.num_vars), a, b)

    def build_loss(self):
        # TODO use MFVI gradient
        # TODO is there a more natural TF way of implementing this?
        q_log_prob = tf.zeros([self.num_minibatch], dtype=tf.float32)
        for i in range(self.q.num_vars):
            q_log_prob += self.q.log_prob_zi(i, self.zs[:, i])

        self.elbo = self.model.log_prob(self.zs) - q_log_prob
        return tf.reduce_mean(q_log_prob * tf.stop_gradient(self.elbo))
