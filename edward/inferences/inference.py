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
from edward.util import check_data, check_latent_vars, get_session, \
    get_variables, Progbar, transform

from tensorflow.contrib.distributions import bijectors


@six.add_metaclass(abc.ABCMeta)
class Inference(object):
  """Abstract base class for inference. All inference algorithms in
  Edward inherit from `Inference`, sharing common methods and
  properties via a class hierarchy.

  Specific algorithms typically inherit from other subclasses of
  `Inference` rather than `Inference` directly. For example, one
  might inherit from the abstract classes `MonteCarlo` or
  `VariationalInference`.

  To build an algorithm inheriting from `Inference`, one must at the
  minimum implement `initialize` and `update`: the former builds
  the computational graph for the algorithm; the latter runs the
  computational graph for the algorithm.

  To reset inference (e.g., internal variable counters incremented
  over training), fetch inference's reset ops from session with
  `sess.run(inference.reset)`.

  #### Examples

  ```python
  # Set up probability model.
  mu = Normal(loc=0.0, scale=1.0)
  x = Normal(loc=mu, scale=1.0, sample_shape=50)

  # Set up posterior approximation.
  qmu_loc = tf.Variable(tf.random_normal([]))
  qmu_scale = tf.nn.softplus(tf.Variable(tf.random_normal([])))
  qmu = Normal(loc=qmu_loc, scale=qmu_scale)

  inference = ed.Inference({mu: qmu}, data={x: tf.zeros(50)})
  ```
  """
  def __init__(self, latent_vars=None, data=None):
    """Create an inference algorithm.

    Args:
      latent_vars: dict.
        Collection of latent variables (of type `RandomVariable` or
        `tf.Tensor`) to perform inference on. Each random variable is
        binded to another random variable; the latter will infer the
        former conditional on data.
      data: dict.
        Data dictionary which binds observed variables (of type
        `RandomVariable` or `tf.Tensor`) to their realizations (of
        type `tf.Tensor`). It can also bind placeholders (of type
        `tf.Tensor`) used in the model to their realizations; and
        prior latent variables (of type `RandomVariable`) to posterior
        latent variables (of type `RandomVariable`).
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
          with tf.variable_scope(None, default_name="data"):
            ph = tf.placeholder(key.dtype, np.shape(value))
            var = tf.Variable(ph, trainable=False, collections=[])
            sess.run(var.initializer, {ph: value})
            self.data[key] = var

  def run(self, variables=None, use_coordinator=True, *args, **kwargs):
    """A simple wrapper to run inference.

    1. Initialize algorithm via `initialize`.
    2. (Optional) Build a TensorFlow summary writer for TensorBoard.
    3. (Optional) Initialize TensorFlow variables.
    4. (Optional) Start queue runners.
    5. Run `update` for `self.n_iter` iterations.
    6. While running, `print_progress`.
    7. Finalize algorithm via `finalize`.
    8. (Optional) Stop queue runners.

    To customize the way inference is run, run these steps
    individually.

    Args:
      variables: list.
        A list of TensorFlow variables to initialize during inference.
        Default is to initialize all variables (this includes
        reinitializing variables that were already initialized). To
        avoid initializing any variables, pass in an empty list.
      use_coordinator: bool.
        Whether to start and stop queue runners during inference using a
        TensorFlow coordinator. For example, queue runners are necessary
        for batch training with file readers.
      *args, **kwargs:
        Passed into `initialize`.
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
  def initialize(self, n_iter=1000, n_print=None, scale=None,
                 auto_transform=True, logdir=None, log_timestamp=True,
                 log_vars=None, debug=False):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Any derived class of `Inference` **must** implement this method.
    No methods which build ops should be called outside `initialize()`.

    Args:
      n_iter: int.
        Number of iterations for algorithm when calling `run()`.
        Alternatively if controlling inference manually, it is the
        expected number of calls to `update()`; this number determines
        tracking information during the print progress.
      n_print: int.
        Number of iterations for each print progress. To suppress print
        progress, then specify 0. Default is `int(n_iter / 100)`.
      scale: dict of RandomVariable to tf.Tensor.
        A tensor to scale computation for any random variable that it is
        binded to. Its shape must be broadcastable; it is multiplied
        element-wise to the random variable. For example, this is useful
        for mini-batch scaling when inferring global variables, or
        applying masks on a random variable.
      auto_transform: bool.
        Whether to automatically transform continuous latent variables
        of unequal support to be on the unconstrained space. It is
        only applied if the argument is `True`, the latent variable
        pair are `ed.RandomVariable`s with the `support` attribute,
        the supports are both continuous and unequal.
      logdir: str.
        Directory where event file will be written. For details,
        see `tf.summary.FileWriter`. Default is to log nothing.
      log_timestamp: bool.
        If True (and `logdir` is specified), create a subdirectory of
        `logdir` to save the specific run results. The subdirectory's
        name is the current UTC timestamp with format 'YYYYMMDD_HHMMSS'.
      log_vars: list.
        Specifies the list of variables to log after each `n_print`
        steps. If None, will log all variables. If `[]`, no variables
        will be logged. `logdir` must be specified for variables to be
        logged.
      debug: bool.
        If True, add checks for `NaN` and `Inf` to all computations
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

    # map from original latent vars to unconstrained versions
    self.transformations = {}
    if auto_transform:
      latent_vars = self.latent_vars.copy()
      # latent_vars maps original latent vars to constrained Q's.
      # latent_vars_unconstrained maps unconstrained vars to unconstrained Q's.
      self.latent_vars = {}
      self.latent_vars_unconstrained = {}
      for z, qz in six.iteritems(latent_vars):
        if hasattr(z, 'support') and hasattr(qz, 'support') and \
                z.support != qz.support and qz.support != 'point':

          # transform z to an unconstrained space
          z_unconstrained = transform(z)
          self.transformations[z] = z_unconstrained

          # make sure we also have a qz that covers the unconstrained space
          if qz.support == "points":
            qz_unconstrained = qz
          else:
            qz_unconstrained = transform(qz)
          self.latent_vars_unconstrained[z_unconstrained] = qz_unconstrained

          # additionally construct the transformation of qz
          # back into the original constrained space
          if z_unconstrained != z:
            qz_constrained = transform(
                qz_unconstrained, bijectors.Invert(z_unconstrained.bijector))

            try:  # attempt to pushforward the params of Empirical distributions
              qz_constrained.params = z_unconstrained.bijector.inverse(
                  qz_unconstrained.params)
            except:  # qz_unconstrained is not an Empirical distribution
              pass

          else:
            qz_constrained = qz_unconstrained

          self.latent_vars[z] = qz_constrained
        else:
          self.latent_vars[z] = qz
          self.latent_vars_unconstrained[z] = qz
      del latent_vars

    if logdir is not None:
      self.logging = True
      if log_timestamp:
        logdir = os.path.expanduser(logdir)
        logdir = os.path.join(
            logdir, datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S"))

      self._summary_key = tf.get_default_graph().unique_name("summaries")
      self._set_log_variables(log_vars)
      self.train_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    else:
      self.logging = False

    self.debug = debug
    if self.debug:
      self.op_check = tf.add_check_numerics_ops()

    # Store reset ops which user can call. Subclasses should append
    # any ops needed to reset internal variables in inference.
    self.reset = [tf.variables_initializer([self.t])]

  @abc.abstractmethod
  def update(self, feed_dict=None):
    """Run one iteration of inference.

    Any derived class of `Inference` **must** implement this method.

    Args:
      feed_dict: dict.
        Feed dictionary for a TensorFlow session run. It is used to feed
        placeholders that are not fed during initialization.

    Returns:
      dict.
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

    Args:
      info_dict: dict.
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

  def _set_log_variables(self, log_vars=None):
    """Log variables to TensorBoard.

    For each variable in `log_vars`, forms a `tf.summary.scalar` if
    the variable has scalar shape; otherwise forms a `tf.summary.histogram`.

    Args:
      log_vars: list.
        Specifies the list of variables to log after each `n_print`
        steps. If None, will log all variables. If `[]`, no variables
        will be logged.
    """
    if log_vars is None:
      log_vars = []
      for key in six.iterkeys(self.data):
        log_vars += get_variables(key)

      for key, value in six.iteritems(self.latent_vars):
        log_vars += get_variables(key)
        log_vars += get_variables(value)

      log_vars = set(log_vars)

    for var in log_vars:
      # replace colons which are an invalid character
      var_name = var.name.replace(':', '/')
      # Log all scalars.
      if len(var.shape) == 0:
        tf.summary.scalar("parameter/{}".format(var_name),
                          var, collections=[self._summary_key])
      elif len(var.shape) == 1 and var.shape[0] == 1:
        tf.summary.scalar("parameter/{}".format(var_name),
                          var[0], collections=[self._summary_key])
      else:
        # If var is multi-dimensional, log a histogram of its values.
        tf.summary.histogram("parameter/{}".format(var_name),
                             var, collections=[self._summary_key])
