from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import numpy as np
import six
import tensorflow as tf

from edward.inferences.inference import Inference
from edward.models import StanModel
from edward.util import get_session

try:
  import prettytensor as pt
except ImportError:
  pass


class VariationalInference(Inference):
  """Base class for variational inference methods.
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
      variables in `latent_vars`' dictionary keys are strings used
      accordingly by the wrapper. `data` is also changed. For
      TensorFlow, Python, and Stan models, the key type is a string;
      for PyMC3, the key type is a Theano shared variable. For
      TensorFlow, Python, and PyMC3 models, the value type is a NumPy
      array or TensorFlow tensor; for Stan, the value type is the type
      according to the Stan program's data block.
    """
    super(VariationalInference, self).__init__(latent_vars, data, model_wrapper)

  def run(self, *args, **kwargs):
    """A simple wrapper to run variational inference.

    1. Initialize via ``initialize``.
    2. Run ``update`` for ``self.n_iter`` iterations.
    3. While running, ``print_progress``.
    4. Finalize via ``finalize``.

    Parameters
    ----------
    *args
      Passed into ``initialize``.
    **kwargs
      Passed into ``initialize``.
    """
    self.initialize(*args, **kwargs)
    for t in range(self.n_iter + 1):
      loss = self.update()
      self.print_progress(t, loss)

    self.finalize()

  def initialize(self, n_iter=1000, n_minibatch=None, n_print=100,
                 optimizer=None, scope=None, logdir=None,
                 use_prettytensor=False):
    """Initialize variational inference algorithm.

    Set up ``tf.train.AdamOptimizer`` with a decaying scale factor.

    Initialize all variables.

    Parameters
    ----------
    n_iter : int, optional
      Number of iterations for optimization.
    n_minibatch : int, optional
      Number of samples for data subsampling. Default is to use
      all the data. Subsampling is available only if all data
      passed in are NumPy arrays and the model is not a Stan
      model. For subsampling details, see
      ``tf.train.slice_input_producer`` and ``tf.train.batch``.
    n_print : int, optional
      Number of iterations for each print progress. To suppress print
      progress, then specify None.
    optimizer : str or tf.train.Optimizer, optional
      A TensorFlow optimizer, to use for optimizing the variational
      objective. Alternatively, one can pass in the name of a
      TensorFlow optimizer, and default parameters for the optimizer
      will be used.
    scope : str, optional
      Scope of TensorFlow variable objects to optimize over.
    logdir : str, optional
      Directory where event file will be written. For details,
      see `tf.train.SummaryWriter`. Default is to write nothing.
    use_prettytensor : bool, optional
      ``True`` if aim to use TensorFlow optimizer or ``False`` if aim
      to use PrettyTensor optimizer (when using PrettyTensor).
      Defaults to TensorFlow.
    """
    self.n_iter = n_iter
    self.n_minibatch = n_minibatch
    self.n_print = n_print
    self.loss = tf.constant(0.0)

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

    if optimizer is None:
      # Use ADAM with a decaying scale factor.
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 0.1
      learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                 global_step,
                                                 100, 0.9, staircase=True)
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif isinstance(optimizer, str):
      if optimizer == 'gradientdescent':
        optimizer = tf.train.GradientDescentOptimizer(0.01)
      elif optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer()
      elif optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(0.01)
      elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
      elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer()
      elif optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(0.01)
      elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(0.01)
      else:
        raise ValueError('Optimizer class not found:', optimizer)

      global_step = None
    elif isinstance(optimizer, tf.train.Optimizer):
      # Custom optimizers have no control over global_step.
      global_step = None
    else:
      raise TypeError()

    loss = self.build_loss()
    if not use_prettytensor:
      var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=scope)
      self.train = optimizer.minimize(loss, global_step=global_step,
                                      var_list=var_list)
    else:
      if scope is not None:
        raise NotImplementedError("PrettyTensor optimizer does not accept "
                                  "a variable scope.")

      # Note PrettyTensor cannot use global_step.
      self.train = pt.apply_optimizer(optimizer, losses=[loss])

    if logdir is not None:
      train_writer = tf.train.SummaryWriter(logdir, tf.get_default_graph())

    init = tf.initialize_all_variables()
    init.run()

    # Start input enqueue threads.
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(coord=self.coord)

  def update(self):
    """Run one iteration of optimizer for variational inference.

    Returns
    -------
    loss : double
      Loss function values after one iteration.
    """
    sess = get_session()
    _, loss = sess.run([self.train, self.loss])
    return loss

  def print_progress(self, t, loss):
    """Print progress to output.

    Parameters
    ----------
    t : int
      Iteration counter.
    loss : double
      Loss function value at iteration ``t``.
    """
    if self.n_print is not None:
      if t % self.n_print == 0:
        print("iter {:d} loss {:.2f}".format(t, loss))
        for rv in six.itervalues(self.latent_vars):
          print(rv)

  def finalize(self):
    """Function to call after convergence.

    Any class based on ``VariationalInference`` **may**
    overwrite this method.
    """
    # Ask threads to stop.
    self.coord.request_stop()
    self.coord.join(self.threads)

  def build_loss(self):
    """Build loss function.

    Empty method.

    Any class based on ``VariationalInference`` **must**
    implement this method.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError()
