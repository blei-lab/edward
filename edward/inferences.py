from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import numpy as np
import six
import tensorflow as tf

from edward.models import StanModel, RandomVariable, Normal, PointMass
from edward.util import copy, get_dims, get_session, hessian, \
    kl_multivariate_normal, log_sum_exp, placeholder

try:
  import prettytensor as pt
except ImportError:
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


class MonteCarlo(Inference):
  """Base class for Monte Carlo inference methods.
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
      array or TensorFlow tensor; for Stan, the value type is the
      type according to the Stan program's data block.
    """
    super(MonteCarlo, self).__init__(latent_vars, data, model_wrapper)


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


class MFVI(VariationalInference):
  """Mean-field variational inference.

  This class implements a variety of "black-box" variational inference
  techniques (Ranganath et al., 2014) that minimize

  .. math::

    KL( q(z; \lambda) || p(z | x) ).

  This is equivalent to maximizing the objective function (Jordan et al., 1999)

  .. math::

    ELBO =  E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ].
  """
  def __init__(self, *args, **kwargs):
    super(MFVI, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, score=None, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    score : bool, optional
      Whether to force inference to use the score function
      gradient estimator. Otherwise default is to use the
      reparameterization gradient if available.
    """
    if score is None and \
       all([rv.is_reparameterized and rv.is_continuous
            for rv in six.itervalues(self.latent_vars)]):
      self.score = False
    else:
      self.score = True

    self.n_samples = n_samples
    return super(MFVI, self).initialize(*args, **kwargs)

  def build_loss(self):
    """Wrapper for the MFVI loss function.

    .. math::

      -ELBO =  -E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

    MFVI supports

    1. score function gradients
    2. reparameterization gradients

    of the loss function.

    If the variational model is a Gaussian distribution, then part of the
    loss function can be computed analytically.

    If the variational model is a normal distribution and the prior is
    standard normal, then part of the loss function can be computed
    analytically following Kingma and Welling (2014),

    .. math::

      E[\log p(x | z) + KL],

    where the KL term is computed analytically.

    Returns
    -------
    result :
      an appropriately selected loss function form
    """
    qz_is_normal = all([isinstance(rv, Normal) for
                       rv in six.itervalues(self.latent_vars)])
    z_is_normal = all([isinstance(rv, Normal) for
                       rv in six.iterkeys(self.latent_vars)])
    is_analytic_kl = qz_is_normal and \
        (z_is_normal or hasattr(self.model_wrapper, 'log_lik'))
    if self.score:
      if is_analytic_kl:
        return self.build_score_loss_kl()
      # Analytic entropies may lead to problems around
      # convergence; for now it is deactivated.
      # elif is_analytic_entropy:
      #    return self.build_score_loss_entropy()
      else:
        return self.build_score_loss()
    else:
      if is_analytic_kl:
        return self.build_reparam_loss_kl()
      # elif is_analytic_entropy:
      #    return self.build_reparam_loss_entropy()
      else:
        return self.build_reparam_loss()

  def build_score_loss(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -ELBO =  -E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

    based on the score function estimator. (Paisley et al., 2012)

    Computed by sampling from :math:`q(z;\lambda)` and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            qz.log_prob(tf.stop_gradient(z_sample[z])))

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_prob[s] = self.model_wrapper.log_prob(x, z_sample)

    p_log_prob = tf.pack(p_log_prob)
    q_log_prob = tf.pack(q_log_prob)

    losses = p_log_prob - q_log_prob
    self.loss = tf.reduce_mean(losses)
    return -tf.reduce_mean(q_log_prob * tf.stop_gradient(losses))

  def build_reparam_loss(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -ELBO =  -E_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]

    based on the reparameterization trick. (Kingma and Welling, 2014)

    Computed by sampling from :math:`q(z;\lambda)` and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(qz.log_prob(z_sample[z]))

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_prob[s] = self.model_wrapper.log_prob(x, z_sample)

    p_log_prob = tf.pack(p_log_prob)
    q_log_prob = tf.pack(q_log_prob)
    self.loss = tf.reduce_mean(p_log_prob - q_log_prob)
    return -self.loss

  def build_score_loss_kl(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -ELBO =  - ( E_{q(z; \lambda)} [ \log p(x | z) ]
             + KL(q(z; \lambda) || p(z)) )

    based on the score function estimator. (Paisley et al., 2012)

    It assumes the KL is analytic.

    For model wrappers, it assumes the prior is :math:`p(z) =
    \mathcal{N}(z; 0, 1)`.

    Computed by sampling from :math:`q(z;\lambda)` and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_lik = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            qz.log_prob(tf.stop_gradient(z_sample[z])))

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_lik[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_lik[s] = self.model_wrapper.log_lik(x, z_sample)

    p_log_lik = tf.pack(p_log_lik)
    q_log_prob = tf.pack(q_log_prob)

    if self.model_wrapper is None:
      kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(
                          qz.mu, qz.sigma, z.mu, z.sigma))
                          for z, qz in six.iteritems(self.latent_vars)])
    else:
      kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(qz.mu, qz.sigma))
                          for qz in six.itervalues(self.latent_vars)])

    self.loss = tf.reduce_mean(p_log_lik) - kl
    return -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_lik)) - kl)

  def build_score_loss_entropy(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -ELBO =  - ( E_{q(z; \lambda)} [ \log p(x, z) ]
            + H(q(z; \lambda)) )

    based on the score function estimator. (Paisley et al., 2012)

    It assumes the entropy is analytic.

    Computed by sampling from :math:`q(z;\lambda)` and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            qz.log_prob(tf.stop_gradient(z_sample[z])))

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_prob[s] = self.model_wrapper.log_prob(x, z_sample)

    p_log_prob = tf.pack(p_log_prob)
    q_log_prob = tf.pack(q_log_prob)

    q_entropy = tf.reduce_sum([qz.entropy()
                               for qz in six.itervalues(self.latent_vars)])

    self.loss = tf.reduce_mean(p_log_prob) + q_entropy
    return -(tf.reduce_mean(q_log_prob * tf.stop_gradient(p_log_prob)) +
             q_entropy)

  def build_reparam_loss_kl(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -ELBO =  - ( E_{q(z; \lambda)} [ \log p(x | z) ]
            + KL(q(z; \lambda) || p(z)) )

    based on the reparameterization trick. (Kingma and Welling, 2014)

    It assumes the KL is analytic.

    For model wrappers, it assumes the prior is :math:`p(z) =
    \mathcal{N}(z; 0, 1)`.

    Computed by sampling from :math:`q(z;\lambda)` and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_lik = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_lik[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_lik[s] = self.model_wrapper.log_lik(x, z_sample)

    p_log_lik = tf.pack(p_log_lik)

    if self.model_wrapper is None:
      kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(
                          qz.mu, qz.sigma, z.mu, z.sigma))
                          for z, qz in six.iteritems(self.latent_vars)])
    else:
      kl = tf.reduce_sum([tf.reduce_sum(kl_multivariate_normal(qz.mu, qz.sigma))
                          for qz in six.itervalues(self.latent_vars)])

    p_log_lik = tf.pack(p_log_lik)
    self.loss = tf.reduce_mean(p_log_lik) - kl
    return -self.loss

  def build_reparam_loss_entropy(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::

      -ELBO =  -( E_{q(z; \lambda)} [ \log p(x , z) ]
            + H(q(z; \lambda)) )

    based on the reparameterization trick. (Kingma and Welling, 2014)

    It assumes the entropy is analytic.

    Computed by sampling from :math:`q(z;\lambda)` and evaluating the
    expectation using Monte Carlo sampling.
    """
    p_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_prob[s] = self.model_wrapper.log_prob(x, z_sample)

    p_log_prob = tf.pack(p_log_prob)

    q_entropy = tf.reduce_sum([qz.entropy()
                               for qz in six.itervalues(self.latent_vars)])

    self.loss = tf.reduce_mean(p_log_prob) + q_entropy
    return -self.loss


class KLpq(VariationalInference):
  """A variational inference method that minimizes the Kullback-Leibler
  divergence from the posterior to the variational model (Cappe et al., 2008)

  .. math::

    KL( p(z |x) || q(z) ).
  """
  def __init__(self, *args, **kwargs):
    super(KLpq, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, *args, **kwargs):
    """Initialization.

    Parameters
    ----------
    n_samples : int, optional
      Number of samples from variational model for calculating
      stochastic gradients.
    """
    self.n_samples = n_samples
    return super(KLpq, self).initialize(*args, **kwargs)

  def build_loss(self):
    """Build loss function. Its automatic differentiation
    is a stochastic gradient of

    .. math::
      KL( p(z |x) || q(z) )
      =
      E_{p(z | x)} [ \log p(z | x) - \log q(z; \lambda) ]

    based on importance sampling.

    Computed as

    .. math::
      1/B \sum_{b=1}^B [ w_{norm}(z^b; \lambda) *
                (\log p(x, z^b) - \log q(z^b; \lambda) ]

    where

    .. math::
      z^b \sim q(z^b; \lambda)

      w_{norm}(z^b; \lambda) = w(z^b; \lambda) / \sum_{b=1}^B (w(z^b; \lambda))

      w(z^b; \lambda) = p(x, z^b) / q(z^b; \lambda)

    which gives a gradient

    .. math::
      - 1/B \sum_{b=1}^B
      w_{norm}(z^b; \lambda) \partial_{\lambda} \log q(z^b; \lambda)

    """
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    for s in range(self.n_samples):
      z_sample = {}
      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope='inference_' + str(s))
        z_sample[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            qz.log_prob(tf.stop_gradient(z_sample[z])))

      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_sample
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      if self.model_wrapper is None:
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope='inference_' + str(s))
          p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(z_sample[z]))

        for x, obs in six.iteritems(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope='inference_' + str(s))
            p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(obs))
      else:
        x = self.data
        p_log_prob[s] = self.model_wrapper.log_prob(x, z_sample)

    p_log_prob = tf.pack(p_log_prob)
    q_log_prob = tf.pack(q_log_prob)

    log_w = p_log_prob - q_log_prob
    log_w_norm = log_w - log_sum_exp(log_w)
    w_norm = tf.exp(log_w_norm)

    self.loss = tf.reduce_mean(w_norm * log_w)
    return -tf.reduce_mean(q_log_prob * tf.stop_gradient(w_norm))


class MAP(VariationalInference):
  """Maximum a posteriori inference.

  We implement this using a ``PointMass`` variational distribution to
  solve the following optimization problem

  .. math::

    \min_{z} - \log p(x,z)
  """
  def __init__(self, latent_vars, data=None, model_wrapper=None):
    """
    Parameters
    ----------
    latent_vars : list of RandomVariable or
                  dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly optimized
      using a ``PointMass`` distribution that is defined
      internally (with support matching each random variable).

    Examples
    --------
    Most explicitly, MAP is specified via a dictionary:

    >>> qpi = PointMass(params=ed.to_simplex(tf.Variable(tf.zeros(K-1))))
    >>> qmu = PointMass(params=tf.Variable(tf.zeros(K*D)))
    >>> qsigma = PointMass(params=tf.nn.softplus(tf.Variable(tf.zeros(K*D))))
    >>> MAP({pi: qpi, mu: qmu, sigma: qsigma}, data)

    We also automate the specification of ``PointMass`` distributions
    (with matching support), so one can pass in a list of latent
    variables instead:

    >>> MAP([beta], {X: np.array(), y: np.array()})
    >>> MAP([pi, mu, sigma], {x: np.array()}

    However, for model wrappers, the list can only have one element:

    >>> MAP(['z'], data, model_wrapper)

    For example, the following is not supported:

    >>> MAP(['pi', 'mu', 'sigma'], data, model_wrapper)

    This is because internally with model wrappers, we have no way
    of knowing the dimensions in which to optimize each
    distribution; further, we do not know their support. For more
    than one random variable, or for constrained support, one must
    explicitly pass in the point mass distributions.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("variational"):
        if model_wrapper is None:
          latent_vars = {rv: PointMass(
              params=tf.Variable(tf.random_normal(rv.batch_shape())))
              for rv in latent_vars}
        elif len(latent_vars) == 1:
          latent_vars = {latent_vars[0]: PointMass(
              params=tf.Variable(
                  tf.squeeze(tf.random_normal([model_wrapper.n_vars]))))}
        elif len(latent_vars) == 0:
          latent_vars = {}
        else:
          raise NotImplementedError("A list of more than one element is "
                                    "not supported. See documentation.")

    super(MAP, self).__init__(latent_vars, data, model_wrapper)

  def build_loss(self):
    """Build loss function. Its automatic differentiation
    is the gradient of

    .. math::
      - \log p(x,z)
    """
    z_mode = {z: qz.value()
              for z, qz in six.iteritems(self.latent_vars)}
    if self.model_wrapper is None:
      p_log_prob = 0.0
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on posterior sample or
      # observed data.
      dict_swap = z_mode
      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          dict_swap[x] = obs

      for z in six.iterkeys(self.latent_vars):
        z_copy = copy(z, dict_swap, scope='inference_' + str(0))
        p_log_prob += tf.reduce_sum(z_copy.log_prob(z_mode[z]))

      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope='inference_' + str(0))
          p_log_prob += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = self.data
      p_log_prob = self.model_wrapper.log_prob(x, z_mode)

    self.loss = p_log_prob
    return -self.loss


class Laplace(MAP):
  """Laplace approximation.

  It approximates the posterior distribution using a normal
  distribution centered at the mode of the posterior.

  We implement this by running ``MAP`` to find the posterior mode.
  This forms the mean of the normal approximation. We then compute
  the Hessian at the mode of the posterior. This forms the
  covariance of the normal approximation.
  """
  def __init__(self, *args, **kwargs):
    super(Laplace, self).__init__(*args, **kwargs)

  def finalize(self):
    """Function to call after convergence.

    Computes the Hessian at the mode.
    """
    # use only a batch of data to estimate hessian
    x = self.data
    z = {z: qz.value() for z, qz in six.iteritems(self.latent_vars)}
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope='variational')
    inv_cov = hessian(self.model_wrapper.log_prob(x, z), var_list)
    print("Precision matrix:")
    print(inv_cov.eval())
    super(Laplace, self).finalize()
