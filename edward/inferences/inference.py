from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf
import os

from datetime import datetime
from edward.models import RandomVariable
from edward.util import get_variables, Progbar
from edward.util import transform as _transform

tfb = tf.contrib.distributions.bijectors


def call_function_up_to_args(f, *args, **kwargs):
  """Call f, removing any args/kwargs it doesn't take as input."""
  import inspect
  if hasattr(f, "_func"):  # tf.make_template()
    argspec = inspect.getargspec(f._func)
  else:
    argspec = inspect.getargspec(f)
  fkwargs = {}
  for k, v in six.iteritems(kwargs):
    if k in argspec.args:
      fkwargs[k] = v
  num_args = len(argspec.args) - len(fkwargs)
  if num_args > 0:
    return f(*args[:num_args], **fkwargs)
  elif len(fkwargs) > 0:
    return f(**fkwargs)
  return f()


def make_intercept(trace, align_data, align_latent, args, kwargs):
  def _intercept(f, *fargs, **fkwargs):
    """Set model's sample values to variational distribution's and data."""
    name = fkwargs.get('name', None)
    key = align_data(name)
    if isinstance(key, int):
      fkwargs['value'] = args[key]
    elif kwargs.get(key, None) is not None:
      fkwargs['value'] = kwargs.get(key)
    elif align_latent(name) is not None:
      qz = trace[align_latent(name)].value
      if isinstance(qz, RandomVariable):
        value = qz.value
      else:  # e.g. replacement is Tensor
        value = tf.convert_to_tensor(qz)
      fkwargs['value'] = value
    # if auto_transform and 'qz' in locals():
    #   # TODO for generation to work, must output original dist. to
    #   keep around TD? must maintain another stack to write to as a
    #   side-effect (or augment the original stack).
    #   return transform(f, qz, *fargs, **fkwargs)
    return f(*fargs, **fkwargs)
  return _intercept


def transform(f, qz, *args, **kwargs):
  """Transform prior -> unconstrained -> q's constraint.

  When using in VI, we keep variational distribution on its original
  space (for sake of implementing only one intercepting function).
  """
  # TODO deal with f or qz being 'point' or 'points'
  if (not hasattr(f, 'support') or not hasattr(qz, 'support') or
          f.support == qz.support):
    return f(*args, **kwargs)
  value = kwargs.pop('value')
  kwargs['value'] = 0.0  # to avoid sampling; TODO follow sample shape
  rv = f(*args, **kwargs)
  # Take shortcuts in logic if p or q are already unconstrained.
  if qz.support in ('real', 'multivariate_real'):
    return _transform(rv, value=value)
  if rv.support in ('real', 'multivariate_real'):
    rv_unconstrained = rv
  else:
    rv_unconstrained = _transform(rv, value=0.0)
  unconstrained_to_constrained = tfb.Invert(_transform(qz).bijector)
  return _transform(rv_unconstrained,
                    unconstrained_to_constrained,
                    value=value)


def train(model, inference=None,
          summary_key=None, n_iter=1000, n_print=None,
          logdir=None, log_timestamp=True,
          variables=None,
          *args, **kwargs):
  """An automated inference engine. It takes a model as input (and
  optional args) and fully trains it until convergence given data to
  return a posterior.

  Given a defaulted inference algorithm (later, we might automate its
  choice, or dynamically apply them), it performs the following steps:

  1. (Optional) Build a TensorFlow summary writer for TensorBoard.
  2. (Optional) Initialize TensorFlow variables.
  3. while not converged: (for now, set by `n_iter` iterations)
    3a. Run update ops.
    3b. If within print window:
      3bi. Print progress.
      3bii. Run convergence diagnostics.
  4. Run finalize (post-training) ops.

  Args:
    n_iter: int, optional.
      Number of iterations for algorithm when calling `run()`.
      Alternatively if controlling inference manually, it is the
      expected number of calls to `update()`; this number determines
      tracking information during the print progress.
    n_print: int, optional.
      Number of iterations for each print progress. To suppress print
      progress, then specify 0. Default is `int(n_iter / 100)`.
    logdir: str, optional.
      Directory where event file will be written. For details,
      see `tf.summary.FileWriter`. Default is to log nothing.
    log_timestamp: bool, optional.
      If True (and `logdir` is specified), create a subdirectory of
      `logdir` to save the specific run results. The subdirectory's
      name is the current UTC timestamp with format 'YYYYMMDD_HHMMSS'.
    variables: list, optional.
      A list of TensorFlow variables to initialize during inference.
      Default is to initialize all variables (this includes
      reinitializing variables that were already initialized). To
      avoid initializing any variables, pass in an empty list.
  """
  if n_print is None:
    n_print = int(n_iter / 100)
  if inference in (bigan_inference, gan_inference, implicit_klqp):
    _update = _gan_update
  elif inference == wgan_inference:
    _update = _wgan_update
  else:
    _update = _default_update
  progbar = Progbar(n_iter)
  t = tf.Variable(0, trainable=False, name="iteration")
  kwargs['t'] = t.assign_add(1)  # add to update()

  if summary_key is not None:
    # TODO _summary_variables()
    summarize = tf.summary.merge_all(key=summary_key)
    if log_timestamp:
      logdir = os.path.expanduser(logdir)
      logdir = os.path.join(
          logdir, datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S"))
    train_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
  else:
    summarize = None
    train_writer = None

  if variables is None:
    init = tf.global_variables_initializer()
  else:
    init = tf.variables_initializer(variables)

  # Feed placeholders in case initialization depends on them.
  feed_dict = kwargs.get('feed_dict', {})
  # TODO use feed dict outside since static
  # feed_dict = {}
  for key, value in six.iteritems(data):
    if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
      feed_dict[key] = value
  init.run(feed_dict)

  for _ in range(n_iter):
    info_dict = _update(progbar, n_print, summarize,
                        train_writer, train_op, *args, **kwargs)

  finalize = None
  if finalize is not None:
    finalize_ops = finalize()
    sess = get_session()
    sess.run(finalize_op, feed_dict)
  else:
    if summary_key is not None:
      train_writer.close()


def _summary_variables(latent_vars=None, data=None, variables=None,
                       *args, **kwargs):
  # Note: to use summary_key, set
  # collections=[tf.get_default_graph().unique_name("summaries")]
  # TODO include in TensorBoard tutorial
  """Log variables to TensorBoard.

  For each variable in `variables`, forms a `tf.summary.scalar` if
  the variable has scalar shape; otherwise forms a `tf.summary.histogram`.

  Args:
    variables: list, optional.
      Specifies the list of variables to log after each `n_print`
      steps. If None, will log all variables. If `[]`, no variables
      will be logged.
  """
  if variables is None:
    variables = []
    for key in six.iterkeys(data):
      variables += get_variables(key)

    for key, value in six.iteritems(latent_vars):
      variables += get_variables(key)
      variables += get_variables(value)

    variables = set(variables)

  for var in variables:
    # replace colons which are an invalid character
    var_name = var.name.replace(':', '/')
    # Log all scalars.
    if len(var.shape) == 0:
      tf.summary.scalar("parameter/{}".format(var_name),
                        var, *args, **kwargs)
    elif len(var.shape) == 1 and var.shape[0] == 1:
      tf.summary.scalar("parameter/{}".format(var_name),
                        var[0], *args, **kwargs)
    else:
      # If var is multi-dimensional, log a histogram of its values.
      tf.summary.histogram("parameter/{}".format(var_name),
                           var, *args, **kwargs)


def _optimize(loss, grads_and_vars, collections=None, var_list=None,
              optimizer=None, use_prettytensor=False, global_step=None):
  """Build optimizer and its train op applied to loss or
  grads_and_vars.

  Args:
    optimizer: str or tf.train.Optimizer, optional.
      A TensorFlow optimizer, to use for optimizing the variational
      objective. Alternatively, one can pass in the name of a
      TensorFlow optimizer, and default parameters for the optimizer
      will be used.
    use_prettytensor: bool, optional.
      `True` if aim to use PrettyTensor optimizer (when using
      PrettyTensor) or `False` if aim to use TensorFlow optimizer.
      Defaults to TensorFlow.
    global_step: tf.Variable, optional.
      A TensorFlow variable to hold the global step.
  """
  if collections is not None:
    # TODO when users call this, this duplicates for GANs
    # train = optimize(loss, grads_and_vars, summary_key)
    # train_d = optimize(loss_d, grads_and_vars_d, summary_key)
    tf.summary.scalar("loss", loss, collections=collections)
    for grad, var in grads_and_vars:
      # replace colons which are an invalid character
      tf.summary.histogram("gradient/" +
                           var.name.replace(':', '/'),
                           grad, collections=collections)
      tf.summary.scalar("gradient_norm/" +
                        var.name.replace(':', '/'),
                        tf.norm(grad), collections=collections)

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
      train_op = optimizer.apply_gradients(grads_and_vars,
                                           global_step=global_step)
    else:
      import prettytensor as pt
      # Note PrettyTensor optimizer does not accept manual updates;
      # it autodiffs the loss directly.
      train_op = pt.apply_optimizer(optimizer, losses=[loss],
                                    global_step=global_step,
                                    var_list=var_list)
  return train_op


def _default_update(progbar, n_print, summarize=None, train_writer=None,
                    *args, **kwargs):
  """Run one iteration of optimization.

  Args:
    args: things like `loss`
    kwargs: things like 'feed_dict'
    feed_dict: dict, optional.
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.

  Returns:
    dict.
    Dictionary of algorithm-specific information. In this case, the
    loss function value after one iteration.
  """
  sess = get_session()
  feed_dict = kwargs.pop('feed_dict', {})
  values = sess.run(list(args) + list(kwargs.values()), feed_dict)
  info_dict = dict(zip(kwargs.keys(), values[len(args):]))

  if n_print != 0:
    t = info_dict['t']
    if t == 1 or t % n_print == 0:
      # TODO do we want specific key names? User can specify whatever
      # in kwargs during run(...).
      # progbar.update(t, {'Loss': info_dict['loss']})
      # progbar.update(t, {'Gen Loss': info_dict['loss'],
      #                    'Disc Loss': info_dict['loss_d']})
      progbar.update(t, {k: v for k, v in six.iteritems(info_dict)
                         if k != 't'})
      if summarize is not None:
        summary = sess.run(summarize, feed_dict)
        train_writer.add_summary(summary, t)

  return info_dict


def _gan_update(train_op, train_op_d, n_print, summarize=None,
                train_writer=None, variables=None, *args, **kwargs):
  """Run one iteration of optimization.

  Args:
    variables: str, optional.
      Which set of variables to update. Either "Disc" or "Gen".
      Default is both.

  Returns:
    dict.
    Dictionary of algorithm-specific information. In this case, the
    iteration number and generative and discriminative losses.

  #### Notes

  The outputted iteration number is the total number of calls to
  `update`. Each update may include updating only a subset of
  parameters.
  """
  # if feed_dict is None:
  #   feed_dict = {}
  # for key, value in six.iteritems(self.data):
  #   if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
  #     feed_dict[key] = value
  sess = get_session()
  feed_dict = kwargs.pop('feed_dict', {})
  if variables is None:
    values = sess.run([train_op, train_op_d] + list(kwargs.values()), feed_dict)
    values = values[2:]
  elif variables == "Gen":
    kwargs['loss_d'] = 0.0
    values = sess.run([train_op] + list(kwargs_temp.values()), feed_dict)
    values = values[1:]
  elif variables == "Disc":
    kwargs['loss'] = 0.0
    values = sess.run([train_op_d] + list(kwargs_temp.values()), feed_dict)
    values = values[1:]
  else:
    raise NotImplementedError("variables must be None, 'Gen', or 'Disc'.")

  if summarize is not None and n_print != 0:
    if t == 1 or t % self.n_print == 0:
      summary = sess.run(summarize, feed_dict)
      train_writer.add_summary(summary, t)

  return dict(zip(kwargs_temp.keys(), values))


def _wgan_update(clip_op, variables=None, *args, **kwargs):
  # TODO make sure increment_t and clipping is called after the update
  # (e.g., with control_dependencies, for monte carlo)
  info_dict = gan_update(variables=variables, *args, **kwargs)

  sess = get_session()
  if clip_op is not None and variables in (None, "Disc"):
    sess.run(clip_op)

  return info_dict
