from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import make_intercept
from edward.models.core import Node, trace


@doc.set_doc(
    args_part_one=(doc.arg_model +
                   doc.arg_align_latent_monte_carlo +
                   doc.arg_align_data +
                   doc.arg_current_state +
                   doc.arg_step_size)[:-1],
    args_part_two=(doc.arg_current_target_log_prob +
                   doc.arg_current_grads_target_log_prob +
                   doc.arg_auto_transform +
                   doc.arg_collections +
                   doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_mcmc_programs=doc.notes_mcmc_programs,
    notes_conditional_inference=doc.notes_conditional_inference)
def sghmc(model,
          align_latent,
          align_data,
          # current_state=None,  # TODO kwarg before arg
          current_state,
          momentum,
          momentum_state,
          learning_rate,
          friction=0.1,
          preconditioner_decay_rate=0.95,
          num_pseudo_batches=1,
          diagonal_bias=1e-8,
          target_log_prob=None,
          grads_target_log_prob=None,
          auto_transform=True,
          collections=None,
          *args, **kwargs):
  """Stochastic gradient Hamiltonian Monte Carlo [@chen2014stochastic].

  SGHMC simulates Hamiltonian dynamics with friction using a discretized
  integrator. Its discretization error goes to zero as the learning
  rate decreases. Namely, it implements the update equations from (15)
  of @chen2014stochastic.

  This function implements an adaptive mass matrix using RMSProp.
  Namely, it uses the update from pre-conditioned SGLD
  [@li2016preconditioned] extended to second-order Langevin dynamics
  (SGHMC): the preconditioner is equal to the inverse of the mass
  matrix [@chen2014stochastic].

  Works for any probabilistic program whose latent variables of
  interest are differentiable. If `auto_transform=True`, the latent
  variables may exist on any constrained differentiable support.

  Args:
  @{args_part_one}
    friction: float.
      Constant scale on the friction term in the Hamiltonian system.
      The implementation may be extended in the future to enable a
      friction per random variable (`friction` would be a callable).
    momentum:
    momentum_state:
    learning_rate:
    friction:
    preconditioner_decay_rate:
    num_pseudo_batches:
    diagonal_bias:
  @{args_part_two}

  Returns:
  @{returns}

  #### Notes

  @{notes_mcmc_programs}

  @{notes_conditional_inference}

  #### Examples

  Consider the following setup.
  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")
  ```
  In graph mode, build `tf.Variable`s which are updated via the Markov
  chain. The update op is fetched at runtime over many iterations.
  ```python
  qmu = tf.get_variable("qmu", initializer=1.)
  qmu_mom = tf.get_variable("qmu_mom", initializer=0.)
  qmu_mom_state = tf.get_variable("qmu_mom_state", initializer=0.)
  new_state, new_momentum, new_momentum_state = ed.sghmc(
      model,
      ...,
      current_state=qmu,
      momentum=qmu_mom,
      momentum_state=qmu_mom_state,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x_data" if name == "x" else None,
      x_data=x_data)
  qmu_update = qmu.assign(new_state)
  qmu_mom_update = qmu_mom.assign(new_momentum)
  qmu_mom_state_update = qmu_mom_state.assign(new_momentum_state)
  ```
  In eager mode, call the function at runtime, updating its inputs
  such as `state`.
  ```python
  qmu = 1.
  qmu_mom = None
  qmu_mom_state = None
  for _ in range(1000):
    new_state, new_momentum, new_momentum_state = ed.sghmc(
        model,
        ...,
        current_state=qmu,
        momentum=qmu_mom,
        momentum_state=qmu_mom_state,
        align_latent=lambda name: "qmu" if name == "mu" else None,
        align_data=lambda name: "x_data" if name == "x" else None,
        x_data=x_data)
    qmu = new_state
    qmu_mom = new_momentum
    qmu_mom_state = new_momentum_state
  ```
  """
  def _target_log_prob_fn(*fargs):
    """Target's unnormalized log-joint density as a function of states."""
    posterior_trace = {state.name.split(':')[0]: Node(arg)
                       for state, arg in zip(states, fargs)}
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    model_trace = trace(model, intercept=intercept, *args, **kwargs)

    p_log_prob = 0.0
    for name, node in six.iteritems(model_trace):
      if align_latent(name) is not None or align_data(name) is not None:
        rv = node.value
        p_log_prob += tf.reduce_sum(rv.log_prob(rv.value))
    return p_log_prob

  is_list_like = lambda x: isinstance(x, (tuple, list))
  maybe_list = lambda x: list(x) if is_list_like(x) else [x]
  states = maybe_list(current_state)

  out = kernel(
      target_log_prob_fn=_target_log_prob_fn,
      current_state=current_state,
      momentum=momentum,
      momentum_state=momentum_state,
      learning_rate=learning_rate,
      frictions=friction,
      preconditioner_decay_rate=preconditioner_decay_rate,
      num_pseudo_batches=num_pseudo_batches,
      diagonal_bias=diagonal_bias,
      current_target_log_prob=target_log_prob,
      current_grads_target_log_prob=grads_target_log_prob)
  return out


def kernel(target_log_prob_fn,
           current_state,
           momentum,
           momentum_state,
           learning_rate,
           frictions=0.1,
           preconditioner_decay_rate=0.95,
           num_pseudo_batches=1,
           diagonal_bias=1e-8,
           current_target_log_prob=None,
           current_grads_target_log_prob=None,
           name=None):
  """Pre-conditioned SGHMC.

  Args:
    ...
    momentum:
    momentum_state: Auxiliary momentums for states (the other is
      momentum for the preconditioner RMSProp.)
    learning_rate: From tf.contrib.bayesflow.SGLDOptimizer.
    frictions:
    preconditioner_decay_rate: From tf.contrib.bayesflow.SGLDOptimizer.
    num_pseudo_batches: From tf.contrib.bayesflow.SGLDOptimizer.
    diagonal_bias: From tf.contrib.bayesflow.SGLDOptimizer.
    ...
  """
  is_list_like = lambda x: isinstance(x, (tuple, list))
  maybe_list = lambda x: list(x) if is_list_like(x) else [x]
  states = maybe_list(current_state)
  momentums = maybe_list(momentum)
  momentums_states = maybe_list(momentum_state)
  with tf.name_scope(name, "sghmc_kernel", states):
    with tf.name_scope("initialize"):
      if current_target_log_prob is None:
        current_target_log_prob = target_log_prob_fn(*states)
      if current_grads_target_log_prob is None:
        current_grads_target_log_prob = tf.gradients(current_target_log_prob, states)

    next_states = []
    next_momentums_states = []
    for state, mom, grad in zip(states, momentums, current_grads_target_log_prob):
      state_update, mom_state_update = _apply_noisy_update(
          mom, grad, learning_rate,
          friction, mom_state,
          diagonal_bias, num_pseudo_batches)
      next_state = state + learning_rate * state_update
      # TODO doesn't this scale the noise incorrectly by additional
      # learning_rate during the update? (same in sgld_optimizer)
      next_mom_state = mom + learning_rate * mom_state_update
      momentum = (mom + (1.0 - preconditioner_decay_rate) *
                  (tf.square(grad) - mom))
      next_states.append(next_state)
      next_momentums_states.append(next_mom_state)
      momentums.append(momentum)

    maybe_flatten = lambda x: x if is_list_like(state) else x[0]
    next_state = maybe_flatten(next_states)
    next_momentum_state = maybe_flatten(next_momentums_states)
    momentum = maybe_flatten(momentums)
    return [
        next_state,
        next_momentum_state,
        momentum,
    ]


def _apply_noisy_update(mom, grad, learning_rate,
                        friction, mom_state,
                        diagonal_bias, num_pseudo_batches):
  """Adapted from tf.contrib.bayesflow.SGLDOptimizer._apply_noisy_update."""
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import random_ops
  stddev = math_ops.cast(math_ops.rsqrt(2 * learning_rate * friction), grad.dtype)
  preconditioner = math_ops.rsqrt(
      mom + math_ops.cast(diagonal_bias, grad.dtype))
  state_update = preconditioner * mom_state
  mom_state_update = (
      -grad * math_ops.cast(num_pseudo_batches,
                            grad.dtype) +
      friction * tf.matmul(preconditioner, mom_state) +
      random_ops.random_normal(array_ops.shape(grad), 1.0, dtype=grad.dtype) *
      stddev)
  return state_update, mom_state_update
