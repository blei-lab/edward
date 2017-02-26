from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import numpy as np
import six
import tensorflow as tf
import warnings

from edward.models import RandomVariable, StanModel
from edward.util import get_session

try:
  import theano
  have_theano = True
except ImportError:
  have_theano = False
  pass


class Inference(object):
  """Base class for Edward inference methods.

  Attributes
  ----------
  latent_vars : dict of RandomVariable to RandomVariable
    Collection of random variables to perform inference on. Each
    random variable is binded to another random variable; the latter
    will infer the former conditional on data.
  data : dict
    Data dictionary whose values may vary at each session run.
  model_wrapper : ed.Model or None
    An optional wrapper for the probability model. If specified, the
    random variables in ``latent_vars``' dictionary keys are strings
    used accordingly by the wrapper.
  """
  def __init__(self, latent_vars=None, data=None, model_wrapper=None):
    """Initialization.

    Parameters
    ----------
    latent_vars : dict of RandomVariable to RandomVariable, optional
      Collection of random variables to perform inference on. Each
      random variable is binded to another random variable; the latter
      will infer the former conditional on data.
    data : dict, optional
      Data dictionary which binds observed variables (of type
      ``RandomVariable`` or ``tf.Tensor``) to their realizations (of
      type ``tf.Tensor``). It can also bind placeholders (of type
      ``tf.Tensor``) used in the model to their realizations; and
      prior latent variables (of type ``RandomVariable``) to posterior
      latent variables (of type ``RandomVariable``).
    model_wrapper : ed.Model, optional
      A wrapper for the probability model. If specified, the random
      variables in ``latent_vars``' dictionary keys are strings
      used accordingly by the wrapper. ``data`` is also changed. For
      TensorFlow, Python, and Stan models, the key type is a string;
      for PyMC3, the key type is a Theano shared variable. For
      TensorFlow, Python, and PyMC3 models, the value type is a NumPy
      array or TensorFlow tensor; for Stan, the value type is the
      type according to the Stan program's data block.

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
      raise TypeError()

    for key, value in six.iteritems(latent_vars):
      if isinstance(value, RandomVariable):
        if isinstance(key, RandomVariable):
          if not key.value().get_shape().is_compatible_with(
                  value.value().get_shape()):
            raise TypeError("Latent variable bindings do not have same shape.")
        elif not isinstance(key, str):
          raise TypeError("Latent variable key has an invalid type.")
      else:
        raise TypeError("Latent variable value has an invalid type.")

    self.latent_vars = latent_vars

    if data is None:
      data = {}
    elif not isinstance(data, dict):
      raise TypeError()

    if isinstance(model_wrapper, StanModel):
      # Stan models do no support data subsampling because they
      # take arbitrary data structure types in the data block
      # and not just NumPy arrays (this makes it unamenable to
      # TensorFlow placeholders). Therefore fix the data
      # dictionary ``self.data`` at compile time to ``data``.
      self.data = data
    else:
      self.data = {}
      for key, value in six.iteritems(data):
        if isinstance(key, RandomVariable):
          if isinstance(value, tf.Tensor):
            if not key.value().get_shape().is_compatible_with(
                    value.get_shape()):
              raise TypeError("Observed variable bindings do not have same "
                              "shape.")

            self.data[key] = tf.cast(value, tf.float32)
          elif isinstance(value, np.ndarray):
            if not key.value().get_shape().is_compatible_with(value.shape):
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
          elif isinstance(value, RandomVariable):
            if not key.value().get_shape().is_compatible_with(
                    value.value().get_shape()):
              raise TypeError("Observed variable bindings do not have same "
                              "shape.")

            self.data[key] = value
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
        elif isinstance(key, str):
          if isinstance(value, tf.Tensor):
            self.data[key] = tf.cast(value, tf.float32)
          elif isinstance(value, np.ndarray):
            ph = tf.placeholder(tf.float32, value.shape)
            var = tf.Variable(ph, trainable=False, collections=[])
            self.data[key] = var
            sess.run(var.initializer, {ph: value})
          else:
            self.data[key] = value
        elif (have_theano and
                isinstance(key, theano.tensor.sharedvar.TensorSharedVariable)):
          self.data[key] = value
        elif isinstance(key, tf.Tensor):
          if isinstance(value, RandomVariable):
            raise TypeError("Data placeholder cannot be bound to a "
                            "RandomVariable.")

          self.data[key] = value
        else:
          raise TypeError("Data key has an invalid type.")

    if model_wrapper is not None:
      warnings.simplefilter('default', DeprecationWarning)
      warnings.warn("Model wrappers are deprecated. Edward is dropping "
                    "support for model wrappers in future versions; use the "
                    "native language instead.", DeprecationWarning)

    self.model_wrapper = model_wrapper

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
      for batch training with the ``n_minibatch`` argument or with
      file readers.
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
      if isinstance(key, tf.Tensor):
        if "Placeholder" in key.op.type:
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

  def initialize(self, n_iter=1000, n_print=None, n_minibatch=None, scale=None,
                 logdir=None, debug=False):
    """Initialize inference algorithm.

    Parameters
    ----------
    n_iter : int, optional
      Number of iterations for algorithm.
    n_print : int, optional
      Number of iterations for each print progress. To suppress print
      progress, then specify 0. Default is ``int(n_iter / 10)``.
    n_minibatch : int, optional
      Number of samples for data subsampling. Default is to use all
      the data. ``n_minibatch`` is available only for TensorFlow,
      Python, and PyMC3 model wrappers; use ``scale`` for Edward's
      language. All data must be passed in as NumPy arrays. For
      subsampling details, see ``tf.train.slice_input_producer`` and
      ``tf.train.batch``.
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
    self.n_minibatch = n_minibatch
    if n_minibatch is not None and \
       self.model_wrapper is not None and \
       not isinstance(self.model_wrapper, StanModel):
      # Re-assign data to batch tensors, with size given by
      # ``n_minibatch``. Don't do this for random variables in data.
      dict_rv = {}
      dict_data = {}
      for key, value in six.iteritems(self.data):
        if isinstance(value, RandomVariable):
          dict_rv[key] = value
        else:
          dict_data[key] = value

      values = list(six.itervalues(dict_data))
      slices = tf.train.slice_input_producer(values)
      # By default use as many threads as CPUs.
      batches = tf.train.batch(slices, n_minibatch,
                               num_threads=multiprocessing.cpu_count())
      if not isinstance(batches, list):
        # ``tf.train.batch`` returns tf.Tensor if ``slices`` is a
        # list of size 1.
        batches = [batches]

      self.data = {key: value for key, value in
                   zip(six.iterkeys(dict_data), batches)}
      self.data.update(dict_rv)

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
      if isinstance(key, tf.Tensor):
        if "Placeholder" in key.op.type:
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
