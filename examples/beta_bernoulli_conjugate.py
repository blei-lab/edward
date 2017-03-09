"""A simple coin flipping example that exploits conjugacy.

Inspired by Stan's toy example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Beta
from edward import conjugacy as conj

ed.set_seed(42)

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
pi = Beta(a=1.0, b=1.0)
x = Bernoulli(p=pi, sample_shape=10)

# COMPLETE CONDITIONAL
pi_cond = conj.complete_conditional(pi, [pi, x])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print('p(pi | x) type:', pi_cond.parameters['name'])
relevant_params = {key: val for key, val in pi_cond.parameters.iteritems()
                   if isinstance(val, tf.Tensor)}
param_vals = sess.run(relevant_params.values(), {x: x_data})
print('parameters:')
for i, j in enumerate(relevant_params.keys()):
    print('%s:\t%.3f' % (j, param_vals[i]))

# print('p(pi | x) parameters:')
# print(pi_cond.parameters)
# for key, val in pi_cond.parameters.iteritems():
#     if isinstance(val, tf.Tensor):
#         print(key, val)

# p_val, x_val = sess.run([p, x], {x: x_data})
# print(p, p_val)
# print(x, x_val)

# # INFERENCE
# qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
# qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
# qp = Beta(a=qp_a, b=qp_b)

# inference = ed.KLqp({p: qp}, data={x: x_data})
# inference.run(n_iter=500)


# import tensorflow as tf
# import edward as ed

# g = tf.Graph()
# with g.as_default():
#     sess = tf.Session()
    
#     pi = ed.models.Beta(0.5, 0.5)
#     flip = ed.models.Bernoulli(p=pi, sample_shape=70)
#     herring = ed.models.Beta(0.5, 0.25, sample_shape=20)

#     pi_post = conj.complete_conditional(pi, [pi, flip, herring])
#     pi_post_samples = pi_post.sample(1000)

# [pi_val, flip_val, temp] = sess.run([pi, flip, pi_post_samples])
# print pi_val
# print flip_val
# hist(temp, 50)
# show()
