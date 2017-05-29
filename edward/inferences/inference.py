from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six
import tensorflow as tf
import os
from datetime import datetime

from edward.models import RandomVariable
from edward.util import check_data, check_latent_vars, get_session, Progbar
from edward.util import get_variables


@six.add_metaclass(abc.ABCMeta)
class Inference(object):
  """Abstract base class for inference. All inference algorithms in
  Edward inherit from ``Inference``, sharing common methods and
  properties via a class hierarchy.

  Specific algorithms typically inherit from other subclasses of
  ``Inference`` rather than ``Inference`` directly. For example, one
  might inherit from the abstract classes ``MonteCarlo`` or
  ``VariationalInference``.

  To build an algorithm inheriting from ``Inference``, one must at the
  minimum implement ``initialize`` and ``update``: the former builds
  the computational graph for the algorithm; the latter runs the
  computational graph for the algorithm.
  """
  def __init__(self, latent_vars=None, data=None):
    """Initialization.

    Parameters
    ----------
    latent_vars : dict, optional
      Collection of latent variables (of type ``RandomVariable`` or
      ``tf.Tensor``) to perform inference on. Each random variable is
      binded to another random variable; the latter will infer the
      former conditional on data.
    data : dict, optional
      Data dictionary which binds observed variables (of type
      ``RandomVariable`` or ``tf.Tensor``) to their realizations (of
      type ``tf.Tensor``). It can also bind placeholders (of type
      ``tf.Tensor``) used in the model to their realizations; and
      prior latent variables (of type ``RandomVariable``) to posterior
      latent variables (of type ``RandomVariable``).

    Examples
    --------
    >>> mu = Normal(loc=tf.constant(0.0), scale=tf.constant(1.0))
    >>> x = Normal(loc=tf.ones(50) * mu, scale=tf.constant(1.0))
    >>>
    >>> qmu_loc = tf.Variable(tf.random_normal([]))
    >>> qmu_scale = tf.nn.softplus(tf.Variable(tf.random_normal([])))
    >>> qmu = Normal(loc=qmu_loc, scale=qmu_scale)
    >>>
    >>> inference = ed.Inference({mu: qmu}, data={x: tf.zeros(50)})
    """
    sess = get_session()
    if latent_vars is None:
      latent_vars = {}
    if data is None:
      data = {}

    check_latent_vars(latent_vars)
    self.latent_vars = latent_vars

    check_data(data)
    self.data = {}
    for key, value in six.iteritems(data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        self.data[key] = value
      elif isinstance(key, (RandomVariable, tf.Tensor)):
        if isinstance(value, (RandomVariable, tf.Tensor)):
          self.data[key] = value
        elif isinstance(value, (float, list, int, np.ndarray, np.number, str)):
          # If value is a Python type, store it in the graph.
          # Assign its placeholder with the key's data type.
          with tf.variable_scope("data"):
            ph = tf.placeholder(key.dtype, np.shape(value))
            var = tf.Variable(ph, trainable=False, collections=[])
            sess.run(var.initializer, {ph: value})
            self.data[key] = var

  def run(self, variables=None, use_coordinator=True, *args, **kwargs):
    """A simple wrapper to run inference.

    1. Initialize algorithm via ``initialize``.
    2. (Optional) Build a TensorFlow summary writer for TensorBoard.
    3. (Optional) Initialize TensorFlow variables.
    4. (Optional) Start queue runners.
    5. Run ``update`` for ``self.n_iter`` iterations.
    6. While running, ``print_progress``.
    7. Finalize algorithm via ``finalize``.
    8. (Optional) Stop queue runners.

    To customize the way inference is run, run these steps
    individually.

    Parameters
    ----------
    variables : list, optional
      A list of TensorFlow variables to initialize during inference.
      Default is to initialize all variables (this includes
      reinitializing variables that were already initialized). To
      avoid initializing any variables, pass in an empty list.
    use_coordinator : bool, optional
      Whether to start and stop queue runners during inference using a
      TensorFlow coordinator. For example, queue runners are necessary
      for batch training with file readers.
    *args
      Passed into ``initialize``.
    **kwargs
      Passed into ``initialize``.
    """
    self.initialize(*args, **kwargs)

    if variables is None:
      init = tf.global_variables_initializer()
    else:
      init = tf.variables_initializer(variables)

    # Feed placeholders in case initialization depends on them.
    feed_dict = {}
    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    init.run(feed_dict)

    if use_coordinator:
      # Start input enqueue threads.
      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(coord=self.coord)

    for _ in range(self.n_iter):
      info_dict = self.update()
      self.print_progress(info_dict)

    self.finalize()

    if use_coordinator:
      # Ask threads to stop.
      self.coord.request_stop()
      self.coord.join(self.threads)

  @abc.abstractmethod
  def initialize(self, n_iter=1000, n_print=None, scale=None, logdir=None,
                 logrun=None, logvars=None, log_max_scalers_per_var=10,
                 debug=False):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computational graph. No ops
    should be created outside the call to ``initialize()``.

    Any derived class of ``Inference`` **must** implement this method.

    Parameters
    ----------
    n_iter : int, optional
      Number of iterations for algorithm.
    n_print : int, optional
      Number of iterations for each print progress. To suppress print
      progress, then specify 0. Default is ``int(n_iter / 100)``.
    scale : dict of RandomVariable to tf.Tensor, optional
      A tensor to scale computation for any random variable that it is
      binded to. Its shape must be broadcastable; it is multiplied
      element-wise to the random variable. For example, this is useful
      for mini-batch scaling when inferring global variables, or
      applying masks on a random variable.
    logdir : str, optional
      Directory where event file will be written. For details,
      see ``tf.summary.FileWriter``. Default is to write nothing.
    logrun : str, optional
      Subdirectory of logdir to save the specific run results, if None
      will set it to current UTC timestamp in the format 'YYYYMMDDTHHMMSS",
      if logrun == '', then the results will be saved to logdir
    logvars : list, optional
      Specifies the list of variables to log after each n_print steps.  If
      None, will log all `latent_variables` that have been given custom names`.
      If logvars == [], no variables will be logged.
    log_max_scalers_per_var : int, default 10
      Enables logging of individual values from 1 dimensional variables up to
      a maximum dimension, if None will log all dimensions.
    debug : bool, optional
      If True, add checks for ``NaN`` and ``Inf`` to all computations
      in the graph. May result in substantially slower execution
      times.
    """
    self.n_iter = n_iter
    if n_print is None:
      self.n_print = int(n_iter / 100)
    else:
      self.n_print = n_print

    self.progbar = Progbar(self.n_iter)
    self.t = tf.Variable(0, trainable=False, name="iteration")

    self.increment_t = self.t.assign_add(1)

    if scale is None:
      scale = {}
    elif not isinstance(scale, dict):
      raise TypeError("scale must be a dict object.")

    self.scale = scale

    if logdir is not None:
      self.logging = True

      if logrun is None:
        # Set default to timestamp
        logrun = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
      if len(logrun):
        logdir = os.path.join(logdir, logrun)

      self.set_log_variables(logvars=logvars,
                             log_max_scalers_per_var=log_max_scalers_per_var)

      self.train_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
      self.summarize = tf.summary.merge_all()
    else:
      self.logging = False

    self.debug = debug
    if self.debug:
      self.op_check = tf.add_check_numerics_ops()

  @abc.abstractmethod
  def update(self, feed_dict=None):
    """Run one iteration of inference.

    Any derived class of ``Inference`` **must** implement this method.

    Parameters
    ----------
    feed_dict : dict, optional
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.

    Returns
    -------
    dict
      Dictionary of algorithm-specific information.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    sess = get_session()
    t = sess.run(self.increment_t)

    if self.debug:
      sess.run(self.op_check, feed_dict)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        summary = sess.run(self.summarize, feed_dict)
        self.train_writer.add_summary(summary, t)

    return {'t': t}

  def print_progress(self, info_dict):
    """Print progress to output.

    Parameters
    ----------
    info_dict : dict
      Dictionary of algorithm-specific information.
    """
    if self.n_print != 0:
      t = info_dict['t']
      if t == 1 or t % self.n_print == 0:
        self.progbar.update(t)

  def finalize(self):
    """Function to call after convergence.
    """
    if self.logging:
      self.train_writer.close()

  def set_log_variables(self, logvars=None, log_max_scalers_per_var=None):
    """Logs variables to TensorBoard.

     For each variable in logvars, creates ``scalar`` and / or ``histogram`` by
     calling ``tf.summary.scalar`` or ``tf.summary.histogram``

     if logvars is None, automatically log all latent variables that have been
     given non-default names.  If logvars is [], no logging will be created.

     Parameters
     ----------
     logvars : list, optional
       A list of variables to be logged
     log_max_scalers_per_var : int, default None
       Enables logging of individual values from 1 dimensional variables up to
       a maximum dimension, if None will log all dimensions.

     Returns
     -------
     None

    """
    if logvars is None:
      logvars = []

      # Add model parameters
      for k in self.data:
        logvars += get_variables(self.data[k])

      # Add model priors
      for k in self.latent_vars:
        logvars += get_variables(self.latent_vars[k])

      # Prune variables to only be custom named variables (without 'Variable')
      # substring
      logvars = [var for var in logvars if 'Variable' not in var.name]

    for var in logvars:
      var_name = var.name.replace(':', '/')  # colons are an invalid character

      # If variable is a one dimensional tensor, log each element in the tensor
      # individually. Only log the first log_max_scalers_per_var variables
      if len(var.shape) == 1:
        for i in range(var.shape[0]):
          if log_max_scalers_per_var is None or i < log_max_scalers_per_var:
            tf.summary.scalar('{}/{}'.format(var_name, i), var[i])

      # If var is multi-dimensional, log the distribution
      if len(var.shape) > 0 and np.max(var.shape) > 1:
        tf.summary.histogram(var_name, var)
