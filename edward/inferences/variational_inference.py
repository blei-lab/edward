from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six
import tensorflow as tf

from edward.inferences.inference import Inference
from edward.models import RandomVariable
from edward.util import get_session, get_variables


@six.add_metaclass(abc.ABCMeta)
class VariationalInference(Inference):
  """Abstract base class for variational inference. Specific
  variational inference methods inherit from `VariationalInference`,
  sharing methods such as a default optimizer.

  To build an algorithm inheriting from `VariationalInference`, one
  must at the minimum implement `build_loss_and_gradients`: it
  determines the loss function and gradients to apply for a given
  optimizer.
  """
  def __init__(self, *args, **kwargs):
    super(VariationalInference, self).__init__(*args, **kwargs)

  def initialize(self, optimizer=None, var_list=None, use_prettytensor=False,
                 global_step=None, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.

    Args:
      optimizer: str or tf.train.Optimizer.
        A TensorFlow optimizer, to use for optimizing the variational
        objective. Alternatively, one can pass in the name of a
        TensorFlow optimizer, and default parameters for the optimizer
        will be used.
      var_list: list of tf.Variable.
        List of TensorFlow variables to optimize over. Default is all
        trainable variables that `latent_vars` and `data` depend on,
        excluding those that are only used in conditionals in `data`.
      use_prettytensor: bool.
        `True` if aim to use PrettyTensor optimizer (when using
        PrettyTensor) or `False` if aim to use TensorFlow optimizer.
        Defaults to TensorFlow.
      global_step: tf.Variable.
        A TensorFlow variable to hold the global step.
    """
    super(VariationalInference, self).initialize(*args, **kwargs)

    if var_list is None:
      # Traverse random variable graphs to get default list of variables.
      var_list = set()
      trainables = tf.trainable_variables()
      for z, qz in six.iteritems(self.latent_vars):
        var_list.update(get_variables(z, collection=trainables))
        var_list.update(get_variables(qz, collection=trainables))

      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable) and \
                not isinstance(qx, RandomVariable):
          var_list.update(get_variables(x, collection=trainables))

      var_list = list(var_list)

    self.loss, grads_and_vars = self.build_loss_and_gradients(var_list)

    if self.logging:
      tf.summary.scalar("loss", self.loss, collections=[self._summary_key])
      for grad, var in grads_and_vars:
        # replace colons which are an invalid character
        tf.summary.histogram("gradient/" +
                             var.name.replace(':', '/'),
                             grad, collections=[self._summary_key])
        tf.summary.scalar("gradient_norm/" +
                          var.name.replace(':', '/'),
                          tf.norm(grad), collections=[self._summary_key])

      self.summarize = tf.summary.merge_all(key=self._summary_key)

    if optimizer is None and global_step is None:
      # Default optimizer always uses a global step variable.
      global_step = tf.Variable(0, trainable=False, name="global_step")

    if isinstance(global_step, tf.Variable):
      starter_learning_rate = 0.1
      learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                 global_step,
                                                 100, 0.9, staircase=True)
    else:
      learning_rate = 0.01

    # Build optimizer.
    if optimizer is None:
      optimizer = tf.train.AdamOptimizer(learning_rate)
    elif isinstance(optimizer, str):
      if optimizer == 'gradientdescent':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      elif optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
      elif optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
      elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
      elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
      elif optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
      elif optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
      else:
        raise ValueError('Optimizer class not found:', optimizer)
    elif not isinstance(optimizer, tf.train.Optimizer):
      raise TypeError("Optimizer must be str, tf.train.Optimizer, or None.")

    with tf.variable_scope(None, default_name="optimizer") as scope:
      if not use_prettytensor:
        self.train = optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)
      else:
        import prettytensor as pt
        # Note PrettyTensor optimizer does not accept manual updates;
        # it autodiffs the loss directly.
        self.train = pt.apply_optimizer(optimizer, losses=[self.loss],
                                        global_step=global_step,
                                        var_list=var_list)

    self.reset.append(tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)))

  def update(self, feed_dict=None):
    """Run one iteration of optimization.

    Args:
      feed_dict: dict.
        Feed dictionary for a TensorFlow session run. It is used to feed
        placeholders that are not fed during initialization.

    Returns:
      dict.
      Dictionary of algorithm-specific information. In this case, the
      loss function value after one iteration.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    sess = get_session()
    _, t, loss = sess.run([self.train, self.increment_t, self.loss], feed_dict)

    if self.debug:
      sess.run(self.op_check, feed_dict)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        summary = sess.run(self.summarize, feed_dict)
        self.train_writer.add_summary(summary, t)

    return {'t': t, 'loss': loss}

  def print_progress(self, info_dict):
    """Print progress to output.
    """
    if self.n_print != 0:
      t = info_dict['t']
      if t == 1 or t % self.n_print == 0:
        self.progbar.update(t, {'Loss': info_dict['loss']})

  @abc.abstractmethod
  def build_loss_and_gradients(self, var_list):
    """Build loss function and its gradients. They will be leveraged
    in an optimizer to update the model and variational parameters.

    Any derived class of `VariationalInference` **must** implement
    this method.

    Raises:
      NotImplementedError.
    """
    raise NotImplementedError()
