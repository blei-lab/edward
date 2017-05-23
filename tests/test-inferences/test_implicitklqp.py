from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal


class test_implicit_klqp_class(tf.test.TestCase):

  def test_normal_run(self):
    def ratio_estimator(data, local_vars, global_vars):
      """Use the optimal ratio estimator, r(z) = log p(z). We add a
      TensorFlow variable as the algorithm assumes that the function
      has parameters to optimize."""
      w = tf.get_variable("w", [])
      return z.log_prob(local_vars[z]) + w

    with self.test_session() as sess:
      z = Normal(loc=5.0, scale=1.0)

      qz = Normal(loc=tf.Variable(tf.random_normal([])),
                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

      inference = ed.ImplicitKLqp({z: qz}, discriminator=ratio_estimator)
      inference.run(n_iter=200)

      self.assertAllClose(qz.mean().eval(), 5.0, atol=1.0)

if __name__ == '__main__':
  ed.set_seed(47324)
  tf.test.main()
