from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.map import MAP
from edward.models import RandomVariable, Normal
from edward.util import copy


class Laplace(MAP):
  """Laplace approximation (Laplace, 1774).

  It approximates the posterior distribution using a normal
  distribution centered at the mode of the posterior.

  We implement this by running ``MAP`` to find the posterior mode.
  This forms the mean of the normal approximation. We then compute
  the Hessian at the mode of the posterior. This forms the
  covariance of the normal approximation.
  """
  def __init__(self, *args, **kwargs):
    super(Laplace, self).__init__(*args, **kwargs)

  def finalize(self):
    """Function to call after convergence.

    Computes the Hessian at the mode.
    """
    # use only a batch of data to estimate hessian
    x = self.data
    z = {z: qz.value() for z, qz in six.iteritems(self.latent_vars)}
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope='posterior')
    inv_cov = tf.hessians(self.model_wrapper.log_prob(x, z), var_list)
    print("Precision matrix:")
    print(inv_cov.eval())
    super(Laplace, self).finalize()
