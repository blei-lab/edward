from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from collections import OrderedDict
from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable
from edward.util import copy, get_session

try:
  from edward.models import Normal, Uniform
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class HMCDA(MonteCarlo):
  """Hamiltonian Monte Carlo with Dual Averaging
  (Hoffman M. and Gelman A., 2014) - Algorithm 5

  Notes
  -----
  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`.
  ``HMC`` substitutes the model's log marginal density

  .. math::

    \log p(x, z) = \log \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
                \\approx \log p(x, z, \\beta^*)

  leveraging a single Monte Carlo sample, where :math:`\\beta^* \sim
  q(\\beta)`. This is unbiased (and therefore asymptotically exact as a
  pseudo-marginal method) if :math:`q(\\beta) = p(\\beta \mid x)`.
  """
  def __init__(self, *args, **kwargs):
    """
    Examples
    --------
    >>> z = Normal(loc=0.0, scale=1.0)
    >>> x = Normal(loc=tf.ones(10) * z, scale=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros(500)))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.HMCDA({z: qz}, data)
    """
    super(HMCDA, self).__init__(*args, **kwargs)

  def initialize(self, n_adapt, delta=0.65, Lambda=0.15, *args, **kwargs):
    """
    Parameters
    ----------
    n_adapt : float
      Number of samples with adaptation for epsilon
    delta : float, optional
      Target accept rate
    Lambda : float, optional
      Target leapfrog length
    """
    # store global scope for log joint calculations
    self._scope = tf.get_default_graph().unique_name("inference") + '/'

    # Find initial epsilon
    step_size = self.find_good_eps()
    sess = get_session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    step_size = sess.run(step_size)

    # Variables for Dual Averaging
    self.epsilon = tf.Variable(step_size, trainable=False)
    self.mu = tf.cast(tf.log(10.0 * step_size), tf.float32)
    self.epsilon_B = tf.Variable(1.0, trainable=False, name="epsilon_bar")
    self.H_B = tf.Variable(0.0, trainable=False, name="H_bar")

    # Parameters for Dual Averaging
    self.n_adapt = n_adapt
    self.delta = tf.constant(delta, dtype=tf.float32)
    self.Lambda = tf.constant(Lambda)

    self.gamma = tf.constant(0.05)
    self.t_0 = tf.constant(10)
    self.kappa = tf.constant(0.75)

    return super(HMCDA, self).initialize(*args, **kwargs)

  def build_update(self):
    """Simulate Hamiltonian dynamics using a numerical integrator.
    Correct for the integrator's discretization error using an
    acceptance ratio. The initial value of epsilon is heuristically chosen
    with Algorithm 4.

    Notes
    -----
    The updates assume each Empirical random variable is directly
    parameterized by ``tf.Variable``s.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}
    old_sample = OrderedDict(old_sample)

    # Sample momentum.
    old_r_sample = OrderedDict()
    for z, qz in six.iteritems(self.latent_vars):
      event_shape = qz.event_shape
      normal = Normal(loc=tf.zeros(event_shape), scale=tf.ones(event_shape))
      old_r_sample[z] = normal.sample()

    # Simulate Hamiltonian dynamics.
    L_m = tf.maximum(1, tf.cast(tf.round(self.Lambda / self.epsilon), tf.int32))
    new_sample, new_r_sample = leapfrog(old_sample, old_r_sample,
                                        self.epsilon, self._log_joint, L_m)

    # Calculate acceptance ratio.
    ratio = kinetic_energy(old_r_sample)
    ratio -= kinetic_energy(new_r_sample)
    ratio += self._log_joint(new_sample)
    ratio -= self._log_joint(old_sample)

    # Accept or reject sample.
    u = Uniform().sample()
    alpha = tf.minimum(1.0, tf.exp(ratio))
    accept = tf.log(u) < ratio

    sample_values = tf.cond(accept, lambda: list(six.itervalues(new_sample)),
                            lambda: list(six.itervalues(old_sample)))
    if not isinstance(sample_values, list):
      # ``tf.cond`` returns tf.Tensor if output is a list of size 1.
      sample_values = [sample_values]

    sample = {z: sample_value for z, sample_value in
              zip(six.iterkeys(new_sample), sample_values)}

    # Use Dual Averaging to adapt epsilon
    should_adapt = self.t <= self.n_adapt
    assign_ops = tf.cond(should_adapt,
                         lambda: self._adapt_step_size(alpha),
                         lambda: self._do_not_adapt_step_size(alpha))

    # Update Empirical random variables.
    for z, qz in six.iteritems(self.latent_vars):
      variable = qz.get_variables()[0]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))

    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(tf.where(accept, 1, 0)))
    return tf.group(*assign_ops)

  def _do_not_adapt_step_size(self, alpha):
    # Do not adapt step size but assign last running averaged epsilon to epsilon
    assign_ops = []
    assign_ops.append(tf.assign(self.H_B, self.H_B).op)
    assign_ops.append(tf.assign(self.epsilon_B, self.epsilon_B).op)
    assign_ops.append(tf.assign(self.epsilon, self.epsilon_B).op)
    return assign_ops

  def _adapt_step_size(self, alpha):
    # Adapt step size as described in Algorithm 5
    assign_ops = []

    factor_H = tf.cast(1 / (self.t + 1 + self.t_0), tf.float32)

    H_B = (1 - factor_H) * self.H_B + factor_H * (self.delta - alpha)
    epsilon = tf.exp(self.mu - tf.sqrt(tf.cast(self.t + 1, tf.float32)) /
                     self.gamma * H_B)

    t_powed = tf.pow(tf.cast(self.t + 1, tf.float32), -self.kappa)
    epsilon_B = tf.exp(t_powed * tf.log(epsilon) +
                       (1 - t_powed) * tf.log(self.epsilon_B))

    # Return ops containing the updates
    assign_ops.append(tf.assign(self.H_B, H_B).op)
    assign_ops.append(tf.assign(self.epsilon, epsilon).op)
    assign_ops.append(tf.assign(self.epsilon_B, epsilon_B).op)
    return assign_ops

  def find_good_eps(self):
    # Heuristically find an inital espilon following Algorithm 4

    # Sample momentum.
    old_r = OrderedDict()

    for z, qz in six.iteritems(self.latent_vars):
      event_shape = qz.event_shape
      normal = Normal(loc=tf.zeros(event_shape), scale=tf.ones(event_shape))
      old_r[z] = normal.sample()

    # Initialize espilon at 1.0
    epsilon = tf.constant(1.0)

    # Calculate log joint probability
    old_z = {z: tf.gather(qz.params, 0)
             for z, qz in six.iteritems(self.latent_vars)}
    old_z = OrderedDict(old_z)

    log_p_joint = -kinetic_energy(old_r)
    log_p_joint += self._log_joint(old_z)

    new_sample, new_r_sample = leapfrog(old_z, old_r,
                                        epsilon, self._log_joint, 1)

    log_p_joint_prime = -kinetic_energy(new_r_sample)
    log_p_joint_prime += self._log_joint(new_sample)

    log_p_joint_diff = log_p_joint_prime - log_p_joint

    # See whether epsilon is too small or to big
    condition = log_p_joint_diff >= tf.log(0.5)
    a = 2.0 * tf.where(condition, 1.0, 0.0) - 1.0

    # Save keys of (z, r) so that we can rebuild the Dict inside the while_loop
    keys_r = list(six.iterkeys(old_r))
    keys_z = list(six.iterkeys(old_z))

    k = tf.constant(0)

    def while_condition(k, _, log_p_joint_prime_loop, log_p_joint, *args):
      accep_big_enough = tf.pow(tf.exp(
          log_p_joint_prime_loop - log_p_joint), a) > tf.pow(2.0, -a)
      to_many_iterations = k < 12
      return tf.logical_and(to_many_iterations, accep_big_enough)

    def body(k, epsilon_loop, _, log_p_joint, values_r, values_z):
        new_epsilon_loop = tf.pow(2.0, a) * epsilon_loop

        # Rebuild the Dicts inside the while_loop since we can only return lists
        old_z_loop = OrderedDict()
        old_r_loop = OrderedDict()
        for i, key in enumerate(values_z):
          old_z_loop[keys_z[i]] = values_z[i]
          old_r_loop[keys_r[i]] = values_r[i]

        new_z_loop, new_r_loop = leapfrog(old_z_loop, old_r_loop,
                                          new_epsilon_loop, self._log_joint, 1)
        new_log_p_joint_prime = -kinetic_energy(new_r_loop)
        new_log_p_joint_prime += self._log_joint(new_z_loop)

        return [k + 1, new_epsilon_loop, new_log_p_joint_prime,
                log_p_joint, values_r, values_z]

    _, new_epsilon, _, _, _, _ = tf.while_loop(
        while_condition, body,
        loop_vars=[k, epsilon, log_p_joint_prime, log_p_joint,
                   list(six.itervalues(old_r.copy())),
                   list(six.itervalues(old_z.copy()))])

    return new_epsilon

  def _log_joint(self, z_sample):
    """Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Parameters
    ----------
    z_sample : dict
      Latent variable keys to samples.
    """
    scope = self._scope + tf.get_default_graph().unique_name("sample")
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    dict_swap = z_sample.copy()
    for x, qx in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    log_joint = 0.0
    for z in six.iterkeys(self.latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      log_joint += tf.reduce_sum(z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(self.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        log_joint += tf.reduce_sum(x_copy.log_prob(dict_swap[x]))

    return log_joint


def leapfrog(z_old, r_old, step_size, log_joint, n_steps):
  # Use a stochastic while_loop since n_steps is a Tensor
  z_new = z_old.copy()
  r_new = r_old.copy()
  keys_z_new = list(six.iterkeys(z_old.copy()))
  first_grad_log_joint = tf.gradients(log_joint(z_new),
                                      list(six.itervalues(z_new)))

  k = tf.constant(0)

  def while_condition(k, v_z_new, v_r_new, grad_log_joint):
     # Stop when k < n_steps
     return k < n_steps

  def body(k, v_z_new, v_r_new, grad_log_joint):
      z_new = OrderedDict()
      for i, key in enumerate(v_z_new):
        z, r = v_z_new[i], v_r_new[i]
        z_new[keys_z_new[i]] = z  # Rebuild the Dict
        v_r_new[i] = r
        v_r_new[i] += 0.5 * step_size * tf.convert_to_tensor(grad_log_joint[i])
        v_z_new[i] = z + step_size * v_r_new[i]

      grad_log_joint = tf.gradients(log_joint(z_new),
                                    list(six.itervalues(z_new)))
      for i, key in enumerate(v_z_new):
        v_r_new[i] += 0.5 * step_size * tf.convert_to_tensor(grad_log_joint[i])
      return [k + 1, v_z_new, v_r_new, grad_log_joint]

  _, v_z_new, v_r_new, _ = tf.while_loop(
      while_condition, body,
      loop_vars=[k, list(six.itervalues(z_new)), list(six.itervalues(r_new)),
                 first_grad_log_joint])

  # Rebuild the Dicts outside the while_loop since we can only pass lists
  for i, key in enumerate(six.iterkeys(z_new)):
    r_new[key] = v_r_new[i]
    z_new[key] = v_z_new[i]

  return z_new, r_new


def kinetic_energy(momentum):
  return tf.reduce_sum([0.5 * tf.reduce_sum(tf.square(r))
                        for r in six.itervalues(momentum)])
