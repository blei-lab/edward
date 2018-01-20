from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.optimizers import KucukelbirOptimizer


class test_sgd_class(tf.test.TestCase):

  def test_kucukelbir_grad(self):
    expected_grads_and_vars = [
      [(3.1018744, 1.0), (1.5509372, 2.0)],
      [(2.7902498, 0.8434107), (1.241244, 1.8959416)],
      [(2.6070995, 0.7563643), (1.0711095, 1.8410041)]
    ]

    x = tf.constant([3., 4., 5.])
    y = tf.constant([.8, .1, .1])
    w1 = tf.Variable(tf.constant(1.))
    w2 = tf.Variable(tf.constant(2.))
    var_list = [w1, w2]

    pred = tf.nn.softmax(x * w1 * w2)
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
    grads = tf.gradients(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))

    optimizer = KucukelbirOptimizer(
      t=0.1,
      delta=10e-3,
      eta=1e-1,
      s_n=tf.Variable([0., 0.]),
      n=tf.Variable(1.)
    )
    train = optimizer.apply_gradients(grads_and_vars)

    actual_grads_and_vars = []

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      for i in range(3):
        actual_grads_and_vars.append(sess.run(grads_and_vars))
        _ = sess.run(train)
        _ = sess.run(optimizer.n.assign_add(1.))

    self.assertAllClose(
      actual_grads_and_vars, expected_grads_and_vars, atol=1e-9)
