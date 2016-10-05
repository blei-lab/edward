from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import numpy as np
import six
import tensorflow as tf

from edward.models import RandomVariable, StanModel
from edward.util import get_session, placeholder


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
    random variables in `latent_vars`' dictionary keys are strings
    used accordingly by the wrapper.
  """
  def __init__(self, latent_vars, data=None, model_wrapper=None):
    """Initialization.

    Parameters
    ----------
    latent_vars : dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on. Each
      random variable is binded to another random variable; the latter
      will infer the former conditional on data.
    data : dict, optional
      Data dictionary which binds observed variables (of type
      `RandomVariable`) to their realizations (of type `tf.Tensor`).
      It can also bind placeholders (of type `tf.Tensor`) used in the
      model to their realizations.
    model_wrapper : ed.Model, optional
      A wrapper for the probability model. If specified, the random
      variables in `latent_vars`' dictionary keys are strings
      used accordingly by the wrapper. `data` is also changed. For
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
    >>> mu = Normal(mu=tf.constant([0.0]), sigma=tf.constant([1.0]))
    >>> x = Normal(mu=tf.ones(N) * mu, sigma=tf.constant([1.0]))
    >>>
    >>> qmu_mu = tf.Variable(tf.random_normal([1]))
    >>> qmu_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([1])))
    >>> qmu = Normal(mu=qmu_mu, sigma=qmu_sigma)
    >>>
    >>> Inference({mu: qmu}, {x: np.array()})
    """
    sess = get_session()
    if not isinstance(latent_vars, dict):
      raise TypeError()

    if data is None:
      data = {}
    elif not isinstance(data, dict):
      raise TypeError()

    self.latent_vars = latent_vars
    self.model_wrapper = model_wrapper

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
        if isinstance(key, RandomVariable) or isinstance(key, str):
          if isinstance(value, tf.Tensor):
            # If ``data`` has TensorFlow placeholders, the user
            # must manually feed them at each step of
            # inference.
            # If ``data`` has tensors that are the output of
            # data readers, then batch training operates
            # according to the reader.
            self.data[key] = tf.cast(value, tf.float32)
          elif isinstance(value, np.ndarray):
            # If ``data`` has NumPy arrays, store the data
            # in the computational graph.
            ph = placeholder(tf.float32, value.shape)
            var = tf.Variable(ph, trainable=False, collections=[])
            self.data[key] = var
            sess.run(var.initializer, {ph: value})
          else:
            raise NotImplementedError()
        else:
          # If key is a placeholder, then don't modify its fed value.
          self.data[key] = value

  def run(self, logdir=None, variables=None, use_coordinator=True,
          *args, **kwargs):
    """A simple wrapper to run inference.

    1. Initialize algorithm via ``initialize``.
    2. (Optional) Build a ``tf.train.SummaryWriter`` for TensorBoard.
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
    logdir : str, optional
      Directory where event file will be written. For details,
      see `tf.train.SummaryWriter`. Default is to write nothing.
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

    if logdir is not None:
      self.train_writer = tf.train.SummaryWriter(logdir, tf.get_default_graph())

    if variables is None:
      init = tf.initialize_all_variables()
    else:
      init = tf.initialize_variables(variables)

    # Feed placeholders in case initialization depends on them.
    feed_dict = {}
    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor):
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

  def initialize(self, n_iter=1000, n_print=None, n_minibatch=None):
    """Initialize inference algorithm.

    Parameters
    ----------
    n_iter : int, optional
      Number of iterations for algorithm.
    n_print : int, optional
      Number of iterations for each print progress. To suppress print
      progress, then specify 0. Default is int(n_iter / 10).
    n_minibatch : int, optional
      Number of samples for data subsampling. Default is to use
      all the data. Subsampling is available only if all data
      passed in are NumPy arrays and the model is not a Stan
      model. For subsampling details, see
      ``tf.train.slice_input_producer`` and ``tf.train.batch``.
    """
    self.n_iter = n_iter
    if n_print is None:
      self.n_print = int(n_iter / 10)
    else:
      self.n_print = n_print

    self.t = tf.Variable(0, trainable=False)
    self.increment_t = self.t.assign_add(1)

    self.n_minibatch = n_minibatch
    if n_minibatch is not None and \
       not isinstance(self.model_wrapper, StanModel):
      # Re-assign data to batch tensors, with size given by
      # ``n_minibatch``.
      values = list(six.itervalues(self.data))
      slices = tf.train.slice_input_producer(values)
      # By default use as many threads as CPUs.
      batches = tf.train.batch(slices, n_minibatch,
                               num_threads=multiprocessing.cpu_count())
      if not isinstance(batches, list):
        # ``tf.train.batch`` returns tf.Tensor if ``slices`` is a
        # list of size 1.
        batches = [batches]

      self.data = {key: value for key, value in
                   zip(six.iterkeys(self.data), batches)}

  def update(self):
    """Run one iteration of inference.

    Returns
    -------
    dict
      Dictionary of algorithm-specific information.
    """
    t = self.increment_t.eval()
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
    pass
