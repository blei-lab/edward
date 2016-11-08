from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf

from edward.models import Normal


class test_inference_scale_class(tf.test.TestCase):

  def test_subgraph(self):
    N = 10
    M = 5
    mu = Normal(mu=0.0, sigma=1.0)
    x = Normal(mu=tf.ones(M) * mu, sigma=tf.ones(M))

    qmu = Normal(mu=tf.Variable(0.0),
                 sigma=tf.constant(1.0))

    x_ph = tf.placeholder(tf.float32, [M])
    data = {x: x_ph}
    inference = ed.KLqp({mu: qmu}, data)
    inference.initialize(scale={x: float(N) / M})
    assert inference.scale[x] == float(N) / M

  def test_minibatch(self):
    N = 10
    M = 5
    mu = Normal(mu=0.0, sigma=1.0)
    x = Normal(mu=tf.ones(N) * mu, sigma=tf.ones(N))

    qmu = Normal(mu=tf.Variable(0.0),
                 sigma=tf.constant(1.0))

    data = {x: tf.zeros(10)}
    inference = ed.KLqp({mu: qmu}, data)
    inference.initialize(n_minibatch=M)
    assert inference.scale[x] == float(N) / M

if __name__ == '__main__':
  tf.test.main()
