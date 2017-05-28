from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import six
import tensorflow as tf

from collections import OrderedDict
from edward.inferences.conjugacy import complete_conditional
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable
from edward.util import check_latent_vars, get_session


class Gibbs(MonteCarlo):
  """Gibbs sampling (Geman and Geman, 1984).
  """
  def __init__(self, latent_vars, proposal_vars=None, data=None):
    """
    Parameters
    ----------
    proposal_vars : dict of RandomVariable to RandomVariable, optional
      Collection of random variables to perform inference on; each is
      binded to its complete conditionals which Gibbs cycles draws on.
      If not specified, default is to use ``ed.complete_conditional``.

    Examples
    --------
    >>> x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    >>>
    >>> p = Beta(1.0, 1.0)
    >>> x = Bernoulli(probs=p, sample_shape=10)
    >>>
    >>> qp = Empirical(tf.Variable(tf.zeros(500)))
    >>> inference = ed.Gibbs({p: qp}, data={x: x_data})
    """
    if proposal_vars is None:
      proposal_vars = {z: complete_conditional(z)
                       for z in six.iterkeys(latent_vars)}
    else:
      check_latent_vars(proposal_vars)

    self.proposal_vars = proposal_vars
    super(Gibbs, self).__init__(latent_vars, data)

  def initialize(self, scan_order='random', *args, **kwargs):
    """
    Parameters
    ----------
    scan_order : list or str, optional
      The scan order for each Gibbs update. If list, it is the
      deterministic order of latent variables. An element in the list
      can be a ``RandomVariable`` or itself a list of
      ``RandomVariable``s (this defines a blocked Gibbs sampler). If
      'random', will use a random order at each update.
    """
    self.scan_order = scan_order
    self.feed_dict = {}
    return super(Gibbs, self).initialize(*args, **kwargs)

  def update(self, feed_dict=None):
    """Run one iteration of Gibbs sampling.

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
    """
    sess = get_session()
    if not self.feed_dict:
      # Initialize feed for all conditionals to be the draws at step 0.
      samples = OrderedDict(self.latent_vars)
      inits = sess.run([qz.params[0] for qz in six.itervalues(samples)])
      for z, init in zip(six.iterkeys(samples), inits):
        self.feed_dict[z] = init

      for key, value in six.iteritems(self.data):
        if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
          self.feed_dict[key] = value
        elif isinstance(key, RandomVariable) and \
                isinstance(value, (tf.Tensor, tf.Variable)):
          self.feed_dict[key] = sess.run(value)

    if feed_dict is None:
      feed_dict = {}

    feed_dict.update(self.feed_dict)

    # Determine scan order.
    if self.scan_order == 'random':
      scan_order = list(six.iterkeys(self.latent_vars))
      random.shuffle(scan_order)
    else:  # list
      scan_order = self.scan_order

    # Fetch samples by iterating over complete conditional draws.
    for z in scan_order:
      if isinstance(z, RandomVariable):
        draw = sess.run(self.proposal_vars[z], feed_dict)
        feed_dict[z] = draw
        self.feed_dict[z] = draw
      else:  # list
        draws = sess.run([self.proposal_vars[zz] for zz in z], feed_dict)
        for zz, draw in zip(z, draws):
          feed_dict[zz] = draw
          self.feed_dict[zz] = draw

    # Assign the samples to the Empirical random variables.
    _, accept_rate = sess.run([self.train, self.n_accept_over_t], feed_dict)
    t = sess.run(self.increment_t)

    if self.debug:
      sess.run(self.op_check, feed_dict)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        summary = sess.run(self.summarize, feed_dict)
        self.train_writer.add_summary(summary, t)

    return {'t': t, 'accept_rate': accept_rate}

  def build_update(self):
    """
    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by ``tf.Variable``s.
    """
    # Update Empirical random variables according to the complete
    # conditionals. We will feed the conditionals when calling ``update()``.
    assign_ops = []
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(
          tf.scatter_update(variable, self.t, self.proposal_vars[z]))

    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)
