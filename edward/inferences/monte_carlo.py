from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.inferences.inference import Inference
from edward.models import Empirical, RandomVariable
from edward.util import get_session


class MonteCarlo(Inference):
  """Base class for Monte Carlo inference methods.
  """
  def __init__(self, latent_vars=None, data=None, model_wrapper=None):
    """Initialization.

    Parameters
    ----------
    latent_vars : list of RandomVariable or
                  dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly approximated
      using a ``Empirical`` random variable that is defined
      internally (with unconstrained support). If dictionary, each
      random variable must be a ``Empirical`` random variable.
    data : dict, optional
      Data dictionary which binds observed variables (of type
      ``RandomVariable`` or ``tf.Tensor``) to their realizations (of
      type ``tf.Tensor``). It can also bind placeholders (of type
      ``tf.Tensor``) used in the model to their realizations.
    model_wrapper : ed.Model, optional
      A wrapper for the probability model. If specified, the random
      variables in ``latent_vars``' dictionary keys are strings used
      accordingly by the wrapper. ``data`` is also changed. For
      TensorFlow, Python, and Stan models, the key type is a string;
      for PyMC3, the key type is a Theano shared variable. For
      TensorFlow, Python, and PyMC3 models, the value type is a NumPy
      array or TensorFlow tensor; for Stan, the value type is the
      type according to the Stan program's data block.

    Examples
    --------
    Most explicitly, ``MonteCarlo`` is specified via a dictionary:

    >>> qpi = Empirical(params=tf.Variable(tf.zeros([T, K-1])))
    >>> qmu = Empirical(params=tf.Variable(tf.zeros([T, K*D])))
    >>> qsigma = Empirical(params=tf.Variable(tf.zeros([T, K*D])))
    >>> MonteCarlo({pi: qpi, mu: qmu, sigma: qsigma}, data)

    The inferred posterior is comprised of ``Empirical`` random
    variables with ``T`` samples. We also automate the specification
    of ``Empirical`` random variables. One can pass in a list of
    latent variables instead:

    >>> MonteCarlo([beta], data)
    >>> MonteCarlo([pi, mu, sigma], data)

    It defaults to ``Empirical`` random variables with 10,000 samples for
    each dimension.

    Notes
    -----
    The number of Monte Carlo iterations is set according to the
    minimum of all ``Empirical`` sizes.

    Initialization is assumed from ``params[0, :]``. This generalizes
    initializing randomly and initializing from user input. Updates
    are along this outer dimension, where iteration t updates
    ``params[t, :]`` in each ``Empirical`` random variable.

    No warm-up is implemented. Users must run MCMC for a long period
    of time, then manually burn in the Empirical random variable.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("posterior"):
        if model_wrapper is None:
          latent_vars = {rv: Empirical(params=tf.Variable(
              tf.zeros([1e4] + rv.get_batch_shape().as_list())))
              for rv in latent_vars}
        else:
          raise NotImplementedError("A list is not supported for model "
                                    "wrappers. See documentation.")
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(qz, Empirical):
          raise TypeError("Posterior approximation must consist of only "
                          "Empirical random variables.")

    super(MonteCarlo, self).__init__(latent_vars, data, model_wrapper)

  def initialize(self, *args, **kwargs):
    kwargs['n_iter'] = np.amin([qz.n for
                                qz in six.itervalues(self.latent_vars)])
    super(MonteCarlo, self).initialize(*args, **kwargs)

    self.n_accept = tf.Variable(0, trainable=False)
    self.n_accept_over_t = self.n_accept / self.t
    self.train = self.build_update()

  def update(self, feed_dict=None):
    """Run one iteration of sampling for Monte Carlo.

    Parameters
    ----------
    feed_dict : dict, optional
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.

    Returns
    -------
    dict
      Dictionary of algorithm-specific information. In this case, the
      acceptance rate of samples since (and including) this iteration.

    Notes
    -----
    We run the increment of ``t`` separately from other ops. Whether the
    others op run with the ``t`` before incrementing or after incrementing
    depends on which is run faster in the TensorFlow graph. Running it
    separately forces a consistent behavior.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor):
        if "Placeholder" in key.op.type:
          feed_dict[key] = value

    sess = get_session()
    _, accept_rate = sess.run([self.train, self.n_accept_over_t], feed_dict)
    t = sess.run(self.increment_t)

    if self.debug:
      sess.run(self.op_check)

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
        accept_rate = info_dict['accept_rate']
        string = 'Iteration {0}'.format(str(t).rjust(len(str(self.n_iter))))
        string += ' [{0}%]'.format(str(int(t / self.n_iter * 100)).rjust(3))
        string += ': Acceptance Rate = {0:.2f}'.format(accept_rate)
        print(string)

  def build_update(self):
    """Build update, which returns an assign op for parameters in
    the Empirical random variables.

    Any derived class of ``MonteCarlo`` **must** implement
    this method.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError()
