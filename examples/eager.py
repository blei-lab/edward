from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Gamma, Normal

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

def model():
  z = Normal(loc=0., scale=1., name='z')
  x = Gamma(tf.nn.softplus(z), 1., sample_shape=1000, name='x')
  return x

def variational():
  qz = Normal(loc=tf.get_variable("loc", shape=[]),
              scale=tf.nn.softplus(tf.get_variable("scale", shape=[])), name='qz')
  return qz

variational = tf.make_template("variational", variational)

x_data = np.random.gamma(5.2, 1.2, size=1000).astype(np.float32)

optimizer = tf.train.AdamOptimizer(1e-2)

# loss, surrogate_loss = ed.klqp(
#     model,
#     variational,
#     align_latent=lambda name: {'z': 'qz'}.get(name),
#     align_data=lambda name: {'x': 'x'}.get(name),
#     x=x_data)
# grads_and_vars = optimizer.compute_gradients(surrogate_loss)
# train_op = optimizer.apply_gradients(grads_and_vars)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for _ in range(2000):
#   sess.run(train_op)

loss_fn = lambda *args: ed.klqp(
    model,
    variational,
    lambda name: {'z': 'qz'}.get(name),
    lambda name: {'x': 'x'}.get(name),
    *args)[1]

value_and_gradients_fn = tfe.implicit_value_and_gradients(loss_fn)

for _ in range(100):
  loss, gradients_and_variables = value_and_gradients_fn(x_data)
  optimizer.apply_gradients(gradients_and_variables)

qz = variational()
print("Posterior mean: {}".format(qz.loc))
print("Posterior variance: {}".format(qz.scale))
