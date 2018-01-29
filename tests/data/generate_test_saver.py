"""Generate `test_saver`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, PointMass


def main(_):
  x_data = np.array([0.0] * 50, dtype=np.float32)

  mu = Normal(loc=0.0, scale=1.0)
  x = Normal(loc=mu, scale=1.0, sample_shape=50)

  with tf.variable_scope("posterior"):
    qmu = PointMass(params=tf.Variable(1.0))

  inference = ed.MAP({mu: qmu}, data={x: x_data})
  inference.run(n_iter=10)

  sess = ed.get_session()
  saver = tf.train.Saver()
  saver.save(sess, "test_saver")

if __name__ == "__main__":
  tf.app.run()
