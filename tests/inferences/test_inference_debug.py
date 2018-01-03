from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_inference_debug_class(tf.test.TestCase):

  def test_placeholder(self):
    with self.test_session():
      N = 5
      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones(N) * mu, scale=tf.ones(N))

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      x_ph = tf.placeholder(tf.float32, [N])
      inference = ed.KLqp({mu: qmu}, data={x: x_ph})
      inference.initialize(debug=True)
      tf.global_variables_initializer().run()
      inference.update(feed_dict={x_ph: np.zeros(N, np.float32)})

  def test_tensor(self):
    with self.test_session():
      N = 5
      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=tf.ones(N) * mu, scale=tf.ones(N))

      qmu = Normal(loc=tf.Variable(0.0), scale=tf.constant(1.0))

      x_data = tf.zeros(N)
      inference = ed.KLqp({mu: qmu}, data={x: x_data})
      inference.run(n_iter=1, debug=True)

if __name__ == '__main__':
  tf.test.main()
