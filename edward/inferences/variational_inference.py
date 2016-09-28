from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import numpy as np
import six
import tensorflow as tf

from edward.inferences.inference import Inference
from edward.models import RandomVariable, StanModel
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

  def initialize(self, n_minibatch=None, optimizer=None, scope=None,
                 use_prettytensor=False, *args, **kwargs):
    """Initialize variational inference algorithm.

    Initialize all variables.

    Parameters
    ----------
    n_minibatch : int, optional
      Number of samples for data subsampling. Default is to use
      all the data. Subsampling is available only if all data
      passed in are NumPy arrays and the model is not a Stan
      model. For subsampling details, see
      ``tf.train.slice_input_producer`` and ``tf.train.batch``.
    optimizer : str or tf.train.Optimizer, optional
      A TensorFlow optimizer, to use for optimizing the variational
      objective. Alternatively, one can pass in the name of a
      TensorFlow optimizer, and default parameters for the optimizer
      will be used.
    scope : str, optional
      Scope of TensorFlow variable objects to optimize over.
    use_prettytensor : bool, optional
      ``True`` if aim to use TensorFlow optimizer or ``False`` if aim
      to use PrettyTensor optimizer (when using PrettyTensor).
      Defaults to TensorFlow.
    """
    super(VariationalInference, self).initialize(*args, **kwargs)
    self.n_minibatch = n_minibatch
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

  def update(self, feed_dict=None):
    """Run one iteration of optimizer for variational inference.

    Parameters
    ----------
    feed_dict : dict, optional
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.

    Returns
    -------
    dict
      Dictionary of algorithm-specific information. In this case, the
      loss function value after one iteration.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if not isinstance(key, RandomVariable) and not isinstance(key, str):
        feed_dict[key] = value

    sess = get_session()
    _, t, loss = sess.run([self.train, self.increment_t, self.loss], feed_dict)
    return {'t': t, 'loss': loss}

  def print_progress(self, info_dict):
    """Print progress to output.
    """
    if self.n_print is not None:
      t = info_dict['t']
      if t == 1 or t % self.n_print == 0:
        loss = info_dict['loss']
        print("iter {:d} loss {:.2f}".format(t, loss))
        for rv in six.itervalues(self.latent_vars):
          print(rv)

  def build_loss(self):
    """Build loss function.

    Any derived class of ``VariationalInference`` **must** implement
    this method.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError()
