"""InverseGamma-Normal with Metropolis-Hastings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import InverseGamma, Normal, Empirical

tf.flags.DEFINE_integer("N", default=1000, help="Number of data points.")
tf.flags.DEFINE_float("loc", default=7.0, help="")
tf.flags.DEFINE_float("scale", default=0.7, help="")

FLAGS = tf.flags.FLAGS


def main(_):
  # Data generation (known mean)
  xn_data = np.random.normal(FLAGS.loc, FLAGS.scale, FLAGS.N)
  print("scale: {}".format(FLAGS.scale))

  # Prior definition
  alpha = 0.5
  beta = 0.7

  # Posterior inference
  # Probabilistic model
  ig = InverseGamma(alpha, beta)
  xn = Normal(FLAGS.loc, tf.sqrt(ig), sample_shape=FLAGS.N)

  # Inference
  qig = Empirical(params=tf.get_variable(
      "qig/params", [1000], initializer=tf.constant_initializer(0.5)))
  proposal_ig = InverseGamma(2.0, 2.0)
  inference = ed.MetropolisHastings({ig: qig},
                                    {ig: proposal_ig}, data={xn: xn_data})
  inference.run()

  sess = ed.get_session()
  print("Inferred scale: {}".format(sess.run(tf.sqrt(qig.mean()))))

if __name__ == "__main__":
  tf.app.run()
