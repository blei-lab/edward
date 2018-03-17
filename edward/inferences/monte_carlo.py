from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six
import tensorflow as tf

from edward.inferences.inference import Inference
from edward.models import Empirical, RandomVariable
from edward.util import get_session


@six.add_metaclass(abc.ABCMeta)
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
  def __init__(self, latent_vars=None, data=None):
    """Create an inference algorithm.

    Args:
      latent_vars: list or dict.
        Collection of random variables (of type `RandomVariable` or
        `tf.Tensor`) to perform inference on. If list, each random
        variable will be approximated using a `Empirical` random
        variable that is defined internally (with unconstrained
        support). If dictionary, each value in the dictionary must be a
        `Empirical` random variable.
      data: dict.
        Data dictionary which binds observed variables (of type
        `RandomVariable` or `tf.Tensor`) to their realizations (of
        type `tf.Tensor`). It can also bind placeholders (of type
        `tf.Tensor`) used in the model to their realizations.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope(None, default_name="posterior"):
        latent_vars = {z: Empirical(params=tf.Variable(tf.zeros(
            [1e4] + z.batch_shape.concatenate(z.event_shape).as_list(),
            dtype=z.dtype)))
            for z in latent_vars}
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(qz, Empirical):
          raise TypeError("Posterior approximation must consist of only "
                          "Empirical random variables.")
        elif len(qz.sample_shape) != 0:
          raise ValueError("Empirical posterior approximations must have "
                           "a scalar sample shape.")

    super(MonteCarlo, self).__init__(latent_vars, data)

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

  def update(self, feed_dict=None):
    """Run one iteration of sampling.

    Args:
      feed_dict: dict.
        Feed dictionary for a TensorFlow session run. It is used to feed
        placeholders that are not fed during initialization.

    Returns:
      dict.
      Dictionary of algorithm-specific information. In this case, the
      acceptance rate of samples since (and including) this iteration.

    #### Notes

    We run the increment of `t` separately from other ops. Whether the
    others op run with the `t` before incrementing or after incrementing
    depends on which is run faster in the TensorFlow graph. Running it
    separately forces a consistent behavior.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    sess = get_session()
    _, accept_rate = sess.run([self.train, self.n_accept_over_t], feed_dict)
    t = sess.run(self.increment_t)

    if self.debug:
      sess.run(self.op_check, feed_dict)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        summary = sess.run(self.summarize, feed_dict)
        self.train_writer.add_summary(summary, t)

    return {'t': t, 'accept_rate': accept_rate}

  def print_progress(self, info_dict):
    """Print progress to output.
    """
    if self.n_print != 0:
      t = info_dict['t']
      if t == 1 or t % self.n_print == 0:
        self.progbar.update(t, {'Acceptance Rate': info_dict['accept_rate']})

  @abc.abstractmethod
  def build_update(self):
    """Build update rules, returning an assign op for parameters in
    the `Empirical` random variables.

    Any derived class of `MonteCarlo` **must** implement this method.

    Raises:
      NotImplementedError.
    """
    raise NotImplementedError()
