"""Normal-normal model using Hamiltonian Monte Carlo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


def model():
  """Normal-Normal with known variance."""
  loc = Normal(loc=0.0, scale=1.0, name="loc")
  x = Normal(loc=loc, scale=1.0, sample_shape=50, name="x")
  return x


def variational():
  qloc = Normal(loc=tf.get_variable("loc", []),
                scale=tf.nn.softplus(tf.get_variable("shape", [])),
                name="qloc")
  return qloc


variational = tf.make_template("variational", variational)

tf.set_random_seed(42)
x_data = np.array([0.0] * 50)

# analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
loss, surrogate_loss = ed.klqp(
    model,
    variational,
    align_latent=lambda name: 'qloc' if name == 'loc' else None,
    align_data=lambda name: 'x_data' if name == 'x' else None,
    x_data=x_data)

optimizer = tf.train.AdamOptimizer(1e-2)
grads_and_vars = optimizer.compute_gradients(surrogate_loss)
train_op = optimizer.apply_gradients(grads_and_vars)

qloc = variational()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
for t in range(1, 5001):
  loss_val, _ = sess.run([loss, train_op])
  if t % 50 == 0:
    mean, stddev = sess.run([qloc.mean(), qloc.stddev()])
    print({"Loss": loss_val,
           "Posterior mean": mean,
           "Posterior stddev": stddev})
