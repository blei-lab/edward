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

try:
  import prettytensor as pt
except ImportError:
  pass


@six.add_metaclass(abc.ABCMeta)
class VariationalInference(Inference):
  """Abstract base class for variational inference. Specific
  variational inference methods inherit from ``VariationalInference``,
  sharing methods such as a default optimizer.

  To build an algorithm inheriting from ``VariaitonalInference``, one
  must at the minimum implement ``build_loss_and_gradients``: it
  determines the loss function and gradients to apply for a given
  optimizer.
  """
  def __init__(self, *args, **kwargs):
    super(VariationalInference, self).__init__(*args, **kwargs)

  def initialize(self, optimizer=None, var_list=None, use_prettytensor=False,
                 *args, **kwargs):
    """Initialize variational inference.

    Parameters
    ----------
    optimizer : str or tf.train.Optimizer, optional
      A TensorFlow optimizer, to use for optimizing the variational
      objective. Alternatively, one can pass in the name of a
      TensorFlow optimizer, and default parameters for the optimizer
      will be used.
    var_list : list of tf.Variable, optional
      List of TensorFlow variables to optimize over. Default is all
      trainable variables that ``latent_vars`` and ``data`` depend on,
      excluding those that are only used in conditionals in ``data``.
    use_prettytensor : bool, optional
      ``True`` if aim to use PrettyTensor optimizer (when using
      PrettyTensor) or ``False`` if aim to use TensorFlow optimizer.
      Defaults to TensorFlow.
    """
    super(VariationalInference, self).initialize(*args, **kwargs)

    latent_var_list = set()
    data_var_list = set()
    if var_list is None:
      # Traverse random variable graphs to get default list of variables.
      trainables = tf.trainable_variables()
      for z, qz in six.iteritems(self.latent_vars):
        if isinstance(z, RandomVariable):
          latent_var_list.update(get_variables(z, collection=trainables))

        latent_var_list.update(get_variables(qz, collection=trainables))

      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable) and \
                not isinstance(qx, RandomVariable):
          data_var_list.update(get_variables(x, collection=trainables))

      var_list = list(data_var_list | latent_var_list)

    self.loss, grads_and_vars = self.build_loss_and_gradients(var_list)

    if self.logging:
      summary_key = 'summaries_' + str(id(self))
      tf.summary.scalar("loss", self.loss, collections=[summary_key])
      for grad, var in grads_and_vars:
        # replace colons which are an invalid character
        tf.summary.histogram("gradient/" +
                             var.name.replace(':', '/'),
                             grad, collections=[summary_key])
        tf.summary.scalar("gradient_norm/" +
                          var.name.replace(':', '/'),
                          tf.norm(grad), collections=[summary_key])

      self.summarize = tf.summary.merge_all(key=summary_key)

    if optimizer is None:
      # Use ADAM with a decaying scale factor.
      global_step = tf.Variable(0, trainable=False, name="global_step")
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
      raise TypeError("Optimizer must be str or tf.train.Optimizer.")

    if not use_prettytensor:
      self.train = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)
    else:
      # Note PrettyTensor optimizer does not accept manual updates;
      # it autodiffs the loss directly.
      self.train = pt.apply_optimizer(optimizer, losses=[self.loss],
                                      global_step=global_step,
                                      var_list=var_list)

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
      if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
        feed_dict[key] = value

    sess = get_session()
    _, t, loss = sess.run([self.train, self.increment_t, self.loss], feed_dict)

    if self.debug:
      sess.run(self.op_check, feed_dict)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        if self.summarize is not None:
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

    Any derived class of ``VariationalInference`` **must** implement
    this method.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError()
