from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import make_log_joint

tfp = tf.contrib.bayesflow


@doc.set_doc(
    args_part_one=(doc.arg_model +
                   doc.arg_align_latent_monte_carlo +
                   doc.arg_align_data +
                   doc.arg_current_state)[:-1],
    args_part_two=(doc.arg_step_size +
                   doc.arg_current_target_log_prob +
                   doc.arg_current_grads_target_log_prob +
                   doc.arg_auto_transform +
                   doc.arg_collections +
                   doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_mcmc_programs=doc.notes_mcmc_programs,
    notes_conditional_inference=doc.notes_conditional_inference)
def hmc(model,
        align_latent,
        align_data,
        current_state=None,
        num_leapfrog_steps=2,
        step_size=0.25,
        current_target_log_prob=None,
        current_grads_target_log_prob=None,
        auto_transform=True,
        collections=None,
        *args, **kwargs):
  """Hamiltonian Monte Carlo, also known as hybrid Monte Carlo
  [@duane1987hybrid; @neal2011mcmc].

  HMC simulates Hamiltonian dynamics using a numerical integrator. The
  integrator has a discretization error and is corrected with a
  Metropolis accept-reject step.

  Works for any probabilistic program whose latent variables of
  interest are differentiable. If `auto_transform=True`, the latent
  variables may exist on any constrained differentiable support.

  Args:
  @{args_part_one}
    num_leapfrog_steps: int.
      Number of steps of numerical integrator.
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
    return x
  ```
  In graph mode, build `tf.Variable`s which are updated via the Markov
  chain. The update op is fetched at runtime over many iterations.
  ```python
  qmu = tf.get_variable("qmu", initializer=1.)
  next_state, _, _ = ed.hmc(
      model,
      ...,
      current_state=qmu,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x_data" if name == "x" else None,
      x_data=x_data)
  qmu_update = qmu.assign(next_state)
  ```
  In eager mode, call the function at runtime, updating its inputs
  such as `current_state`.
  ```python
  qmu = 1.
  next_log_prob = None
  next_gradients = None
  for _ in range(1000):
    next_state, next_log_prob, next_gradients = ed.hmc(
        model,
        ...,
        current_state=qmu,
        align_latent=lambda name: "qmu" if name == "mu" else None,
        align_data=lambda name: "x_data" if name == "x" else None,
        current_target_log_prob=next_log_prob,
        current_grads_target_log_prob=next_gradients,
        x_data=x_data)
    qmu = next_state
  ```
  """
  out = tfp.hmc.kernel(
      target_log_prob_fn=make_log_joint(model, current_state),
      current_state=current_state,
      step_size=step_size,
      num_leapfrog_steps=num_leapfrog_steps,
      current_target_log_prob=current_target_log_prob,
      current_grads_target_log_prob=current_grads_target_log_prob)
  return out
