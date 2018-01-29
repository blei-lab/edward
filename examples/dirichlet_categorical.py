"""Dirichlet-Categorical with variational inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Dirichlet

tf.flags.DEFINE_integer("N", default=1000, help="")
tf.flags.DEFINE_integer("K", default=4, help="")

FLAGS = tf.flags.FLAGS


def main(_):
  # DATA
  pi_true = np.random.dirichlet(np.array([20.0, 30.0, 10.0, 10.0]))
  z_data = np.array([np.random.choice(FLAGS.K, 1, p=pi_true)[0]
                     for n in range(FLAGS.N)])
  print("pi: {}".format(pi_true))

  # MODEL
  pi = Dirichlet(tf.ones(4))
  z = Categorical(probs=pi, sample_shape=FLAGS.N)

  # INFERENCE
  qpi = Dirichlet(tf.nn.softplus(
      tf.get_variable("qpi/concentration", [FLAGS.K])))

  inference = ed.KLqp({pi: qpi}, data={z: z_data})
  inference.run(n_iter=1500, n_samples=30)

  sess = ed.get_session()
  print("Inferred pi: {}".format(sess.run(qpi.mean())))

if __name__ == "__main__":
  tf.app.run()
