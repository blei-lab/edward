from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def alp_optimizer_apply_gradients(n, s_n, grads_and_vars):
  ops = []
  for i, (grad, var) in enumerate(grads_and_vars):
    updated_s_n = s_n[i].assign( (t * grad**2) + (1 - t) * s_n[i] )

    p_n_first = eta * n**(-.5 + delta)
    p_n_second = (1 + tf.sqrt(updated_s_n[i]))**(-1)
    p_n = p_n_first * p_n_second

    updated_var = var.assign_add(-p_n * grad)
    ops.append(updated_var)
  return ops
