from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import RandomVariable
from edward.util import get_session, Progbar


class Inference(object):
  """Base class for Edward inference methods.
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
    >>> mu = Normal(mu=tf.constant(0.0), sigma=tf.constant(1.0))
    >>> x = Normal(mu=tf.ones(50) * mu, sigma=tf.constant(1.0))
    >>>
    >>> qmu_mu = tf.Variable(tf.random_normal([]))
    >>> qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([])))
    >>> qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)
    >>>
    >>> inference = ed.Inference({mu: qmu}, data={x: tf.zeros(50)})
    """
    sess = get_session()
    if latent_vars is None:
      latent_vars = {}
    elif not isinstance(latent_vars, dict):
      raise TypeError("latent_vars must have type dict.")

    for key, value in six.iteritems(latent_vars):
      if not isinstance(key, (RandomVariable, tf.Tensor)):
        raise TypeError("Latent variable key has an invalid type: "
                        "{}".format(type(key)))
      elif not isinstance(value, (RandomVariable, tf.Tensor)):
        raise TypeError("Latent variable value has an invalid type: "
                        "{}".format(type(value)))
      elif not key.get_shape().is_compatible_with(value.get_shape()):
        raise TypeError("Key-value pair in latent_vars does not have same "
                        "shape: {}, {}".format(key.get_shape(),
                                               value.get_shape()))
      elif key.dtype != value.dtype:
        raise TypeError("Key-value pair in latent_vars does not have same "
                        "dtype: {}, {}".format(key.dtype, value.dtype))

    self.latent_vars = latent_vars

    if data is None:
      data = {}
    elif not isinstance(data, dict):
      raise TypeError("data must have type dict.")

    self.data = {}
    for key, value in six.iteritems(data):
      if isinstance(key, RandomVariable) or \
         (isinstance(key, tf.Tensor) and "Placeholder" not in key.op.type):
        if isinstance(value, (tf.Tensor, RandomVariable)):
          if not key.get_shape().is_compatible_with(value.get_shape()):
            raise TypeError("Key-value pair in data does not have same "
                            "shape: {}, {}".format(key.get_shape(),
                                                   value.get_shape()))
          elif key.dtype != value.dtype:
            raise TypeError("Key-value pair in data does not have same "
                            "dtype: {}, {}".format(key.dtype, value.dtype))

          self.data[key] = value
        elif isinstance(value, (list, np.ndarray, np.number)):
          if not key.get_shape().is_compatible_with(np.shape(value)):
            raise TypeError("Key-value pair in data does not have same "
                            "shape: {}, {}".format(key.get_shape(),
                                                   np.shape(value)))
          elif not isinstance(value, list) and \
                  not np.issubdtype(value.dtype, np.float) and \
                  not np.issubdtype(value.dtype, np.int) and \
                  not np.issubdtype(value.dtype, np.str):
            raise TypeError("Data value has an invalid dtype: "
                            "{}".format(value.dtype))

          # If value is a list or np.ndarray, store it in the graph.
          # Assign its placeholder with the key's data type.
          ph = tf.placeholder(key.dtype, np.shape(value))
          var = tf.Variable(ph, trainable=False, collections=[])
          sess.run(var.initializer, {ph: value})
          self.data[key] = var
        elif isinstance(value, (float, int, str)):
          ph = tf.placeholder(key.dtype, ())
          var = tf.Variable(ph, trainable=False, collections=[])
          sess.run(var.initializer, {ph: value})
          self.data[key] = var
        else:
          raise TypeError("Data value has an invalid type: "
                          "{}".format(type(value)))
      elif isinstance(key, tf.Tensor):
        if isinstance(value, RandomVariable):
          raise TypeError("The value of a feed cannot be a ed.RandomVariable "
                          "object. "
                          "Acceptable feed values include Python scalars, "
                          "strings, lists, numpy ndarrays, or TensorHandles.")
        elif isinstance(value, tf.Tensor):
          raise TypeError("The value of a feed cannot be a tf.Tensor object. "
                          "Acceptable feed values include Python scalars, "
                          "strings, lists, numpy ndarrays, or TensorHandles.")

        self.data[key] = value
      else:
        raise TypeError("Data key has an invalid type: {}".format(type(key)))

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

  def initialize(self, n_iter=1000, n_print=None, scale=None, logdir=None,
                 debug=False):
    """Initialize inference algorithm.

    Parameters
    ----------
    n_iter : int, optional
      Number of iterations for algorithm.
    n_print : int, optional
      Number of iterations for each print progress. To suppress print
      progress, then specify 0. Default is ``int(n_iter / 10)``.
    scale : dict of RandomVariable to tf.Tensor, optional
      A tensor to scale computation for any random variable that it is
      binded to. Its shape must be broadcastable; it is multiplied
      element-wise to the random variable. For example, this is useful
      for mini-batch scaling when inferring global variables, or
      applying masks on a random variable.
    logdir : str, optional
      Directory where event file will be written. For details,
      see ``tf.summary.FileWriter``. Default is to write nothing.
    debug : bool, optional
      If True, add checks for ``NaN`` and ``Inf`` to all computations
      in the graph. May result in substantially slower execution
      times.
    """
    self.n_iter = n_iter
    if n_print is None:
      self.n_print = int(n_iter / 10)
    else:
      self.n_print = n_print

    self.progbar = Progbar(self.n_iter)
    self.t = tf.Variable(0, trainable=False)
    self.increment_t = self.t.assign_add(1)

    if scale is None:
      scale = {}
    elif not isinstance(scale, dict):
      raise TypeError()

    self.scale = scale

    if logdir is not None:
      self.logging = True
      self.train_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
      self.summarize = tf.summary.merge_all()
    else:
      self.logging = False

    self.debug = debug
    if self.debug:
      self.op_check = tf.add_check_numerics_ops()

  def update(self, feed_dict=None):
    """Run one iteration of inference.

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
      sess.run(self.op_check)

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
