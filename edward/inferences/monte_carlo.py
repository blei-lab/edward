from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import Empirical, RandomVariable
from edward.util import get_session


class MonteCarlo(Inference):
  """Abstract base class for Monte Carlo. Specific Monte Carlo methods
  inherit from `MonteCarlo`, sharing methods in this class.

  To build an algorithm inheriting from `MonteCarlo`, one must at the
  minimum implement `build_update`: it determines how to assign
  the samples in the `Empirical` approximations.

  #### Notes

  The number of Monte Carlo iterations is set according to the
  minimum of all `Empirical` sizes.

  Initialization is assumed from `params[0, :]`. This generalizes
  initializing randomly and initializing from user input. Updates
  are along this outer dimension, where iteration t updates
  `params[t, :]` in each `Empirical` random variable.

  No warm-up is implemented. Users must run MCMC for a long period
  of time, then manually burn in the Empirical random variable.

  #### Examples

  Most explicitly, `MonteCarlo` is specified via a dictionary:

  ```python
  qpi = Empirical(params=tf.Variable(tf.zeros([T, K-1])))
  qmu = Empirical(params=tf.Variable(tf.zeros([T, K*D])))
  qsigma = Empirical(params=tf.Variable(tf.zeros([T, K*D])))
  ed.MonteCarlo({pi: qpi, mu: qmu, sigma: qsigma}, data)
  ```

  The inferred posterior is comprised of `Empirical` random
  variables with `T` samples. We also automate the specification
  of `Empirical` random variables. One can pass in a list of
  latent variables instead:

  ```python
  ed.MonteCarlo([beta], data)
  ed.MonteCarlo([pi, mu, sigma], data)
  ```

  It defaults to `Empirical` random variables with 10,000 samples for
  each dimension.
  """
  """Create an inference algorithm.

  Args:
    latent_vars: list or dict, optional.
      Collection of random variables (of type `RandomVariable` or
      `tf.Tensor`) to perform inference on. If list, each random
      variable will be approximated using a `Empirical` random
      variable that is defined internally (with unconstrained
      support). If dictionary, each value in the dictionary must be a
      `Empirical` random variable.
    data: dict, optional.
      Data dictionary which binds observed variables (of type
      `RandomVariable` or `tf.Tensor`) to their realizations (of
      type `tf.Tensor`). It can also bind placeholders (of type
      `tf.Tensor`) used in the model to their realizations.
  """
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars = {z: Empirical(params=tf.Variable(tf.zeros(
          [1e4] + z.batch_shape.concatenate(z.event_shape).as_list())))
          for z in latent_vars}
  elif isinstance(latent_vars, dict):
    for qz in six.itervalues(latent_vars):
      if not isinstance(qz, Empirical):
        raise TypeError("Posterior approximation must consist of only "
                        "Empirical random variables.")
      elif len(qz.sample_shape) != 0:
        raise ValueError("Empirical posterior approximations must have "
                         "a scalar sample shape.")

  def initialize(self, *args, **kwargs):
    kwargs['n_iter'] = np.amin([qz.params.shape.as_list()[0] for
                                qz in six.itervalues(self.latent_vars)])
    super(MonteCarlo, self).initialize(*args, **kwargs)

    self.n_accept = tf.Variable(0, trainable=False, name="n_accept")
    self.n_accept_over_t = self.n_accept / self.t
    self.train = self.build_update()

    self.reset.append(tf.variables_initializer([self.n_accept]))

    if self.logging:
      tf.summary.scalar("n_accept", self.n_accept,
                        collections=[self._summary_key])
      self.summarize = tf.summary.merge_all(key=self._summary_key)
