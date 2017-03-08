from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models import RandomVariable
from edward.util import get_session


class Inference(object):
  """Base class for Edward inference methods.

  Attributes
  ----------
  latent_vars : dict
    Collection of latent variables (of type ``RandomVariable`` or
    ``tf.Tensor``) to perform inference on. Each random variable is
    binded to another random variable; the latter will infer the
    former conditional on data.
  data : dict
    Data dictionary which binds observed variables (of type
    ``RandomVariable`` or ``tf.Tensor``) to their realizations (of
    type ``tf.Tensor``).
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

    Notes
    -----
    If ``data`` is not passed in, the dictionary is empty.

    Three options are available for batch training:

    1. internally if user passes in data as a dictionary of NumPy
       arrays;
    2. externally if user passes in data as a dictionary of
       TensorFlow placeholders (and manually feeds them);
    3. externally if user passes in data as TensorFlow tensors
       which are the outputs of data readers.

    Examples
    --------
    >>> mu = Normal(mu=tf.constant(0.0), sigma=tf.constant(1.0))
    >>> x = Normal(mu=tf.ones(N) * mu, sigma=tf.constant(1.0))
    >>>
    >>> qmu_mu = tf.Variable(tf.random_normal([1]))
    >>> qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([1])))
    >>> qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)
    >>>
    >>> Inference({mu: qmu}, {x: tf.constant([0.0] * N)})
    """
    sess = get_session()
    if latent_vars is None:
      latent_vars = {}
    elif not isinstance(latent_vars, dict):
      raise TypeError("latent_vars must have type dict.")

    for key, value in six.iteritems(latent_vars):
      if not isinstance(key, (RandomVariable, tf.Tensor)):
        raise TypeError("Latent variable key has an invalid type.")
      elif not isinstance(value, (RandomVariable, tf.Tensor)):
        raise TypeError("Latent variable value has an invalid type.")
      elif not key.get_shape().is_compatible_with(value.get_shape()):
        raise TypeError("Latent variable bindings do not have same shape.")

    self.latent_vars = latent_vars

    if data is None:
      data = {}
    elif not isinstance(data, dict):
      raise TypeError("data must have type dict.")

    self.data = {}
    for key, value in six.iteritems(data):
      if isinstance(key, RandomVariable) or \
         (isinstance(key, tf.Tensor) and "Placeholder" not in key.op.type):
        if isinstance(value, tf.Tensor):
          if not key.get_shape().is_compatible_with(value.get_shape()):
            raise TypeError("Observed variable bindings do not have same "
                            "shape.")

          self.data[key] = tf.cast(value, tf.float32)
        elif isinstance(value, RandomVariable):
          if not key.get_shape().is_compatible_with(value.get_shape()):
            raise TypeError("Observed variable bindings do not have same "
                            "shape.")

          self.data[key] = value
        elif isinstance(value, np.ndarray):
          if not key.get_shape().is_compatible_with(value.shape):
            raise TypeError("Observed variable bindings do not have same "
                            "shape.")

          # If value is a np.ndarray, store it in the graph. Assign its
          # placeholder to an appropriate data type.
          if np.issubdtype(value.dtype, np.float):
            ph_type = tf.float32
          elif np.issubdtype(value.dtype, np.int):
            ph_type = tf.int32
          else:
            raise TypeError("Data value has an unsupported type.")
          ph = tf.placeholder(ph_type, value.shape)
          var = tf.Variable(ph, trainable=False, collections=[])
          self.data[key] = var
          sess.run(var.initializer, {ph: value})
        elif isinstance(value, np.number):
          if np.issubdtype(value.dtype, np.float):
            ph_type = tf.float32
          elif np.issubdtype(value.dtype, np.int):
            ph_type = tf.int32
          else:
              raise TypeError("Data value as an invalid type.")
          ph = tf.placeholder(ph_type, value.shape)
          var = tf.Variable(ph, trainable=False, collections=[])
          self.data[key] = var
          sess.run(var.initializer, {ph: value})
        elif isinstance(value, float):
          ph_type = tf.float32
          ph = tf.placeholder(ph_type, ())
          var = tf.Variable(ph, trainable=False, collections=[])
          self.data[key] = var
          sess.run(var.initializer, {ph: value})
        elif isinstance(value, int):
          ph_type = tf.int32
          ph = tf.placeholder(ph_type, ())
          var = tf.Variable(ph, trainable=False, collections=[])
          self.data[key] = var
          # handle if value is `bool` which this case catches
          sess.run(var.initializer, {ph: int(value)})
        else:
          raise TypeError("Data value has an invalid type.")
      elif isinstance(key, tf.Tensor):
        if isinstance(value, RandomVariable):
          raise TypeError("Data placeholder cannot be bound to a "
                          "RandomVariable.")

        self.data[key] = value
      else:
        raise TypeError("Data key has an invalid type.")

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
        string = 'Iteration {0}'.format(str(t).rjust(len(str(self.n_iter))))
        string += ' [{0}%]'.format(str(int(t / self.n_iter * 100)).rjust(3))
        print(string)

  def finalize(self):
    """Function to call after convergence.
    """
    if self.logging:
      self.train_writer.close()
