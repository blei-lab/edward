from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import make_intercept
from edward.models.core import Node, trace

tfp = tf.contrib.bayesflow


@doc.set_doc(
    args_part_one=(doc.arg_model +
                   doc.arg_align_latent_monte_carlo +
                   doc.arg_align_data +
                   doc.arg_current_state)[:-1],
    args_part_two=(doc.arg_current_target_log_prob +
                   doc.arg_current_grads_target_log_prob +
                   doc.arg_auto_transform +
                   doc.arg_collections +
                   doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_mcmc_programs=doc.notes_mcmc_programs,
    notes_conditional_inference=doc.notes_conditional_inference)
def sgld(model,
         align_latent,
         align_data,
         # current_state=None,  # TODO kwarg before arg
         current_state,
         momentum,
         learning_rate,
         preconditioner_decay_rate=0.95,
         num_pseudo_batches=1,
         diagonal_bias=1e-8,
         target_log_prob=None,
         grads_target_log_prob=None,
         auto_transform=True,
         collections=None,
         *args, **kwargs):
  """Stochastic gradient Langevin dynamics [@welling2011bayesian].

  SGLD simulates Langevin dynamics using a discretized integrator. Its
  discretization error goes to zero as the learning rate decreases.

  This function implements an adaptive preconditioner using RMSProp
  [@li2016preconditioned].

  Works for any probabilistic program whose latent variables of
  interest are differentiable. If `auto_transform=True`, the latent
  variables may exist on any constrained differentiable support.

  Args:
  @{args_part_one}
    momentum:
    learning_rate:
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
  new_state, new_momentum = ed.sgld(
      model,
      ...,
      current_state=qmu,
      momentum=qmu_mom,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x_data" if name == "x" else None,
      x_data=x_data)
  qmu_update = qmu.assign(new_state)
  qmu_mom_update = qmu_mom.assign(new_momentum)
  ```
  In eager mode, call the function at runtime, updating its inputs
  such as `state`.
  ```python
  qmu = 1.
  qmu_mom = None
  for _ in range(1000):
    new_state, momentum = ed.sgld(
        model,
        ...,
        current_state=qmu,
        momentum=qmu_mom,
        align_latent=lambda name: "qmu" if name == "mu" else None,
        align_data=lambda name: "x_data" if name == "x" else None,
        x_data=x_data)
    qmu = new_state
    qmu_mom = new_momentum
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
  states = maybe_list(state)

  out = tfp.sgld.kernel(
      target_log_prob_fn=_target_log_prob_fn,
      current_state=state,
      momentum=momentum,
      learning_rate=learning_rate,
      preconditioner_decay_rate=preconditioner_decay_rate,
      num_pseudo_batches=num_pseudo_batches,
      diagonal_bias=diagonal_bias,
      current_target_log_prob=target_log_prob,
      current_grads_target_log_prob=grads_target_log_prob)
  return out
