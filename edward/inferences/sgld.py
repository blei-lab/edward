from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)
from edward.models.core import Node, Trace

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
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

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


def kernel(target_log_prob_fn,
           current_state,
           momentum,
           learning_rate,
           preconditioner_decay_rate=0.95,
           num_pseudo_batches=1,
           diagonal_bias=1e-8,
           current_target_log_prob=None,
           current_grads_target_log_prob=None,
           name=None):
  """Runs the stochastic gradient Langevin dynamics transition kernel.

  This implements the preconditioned Stochastic Gradient Langevin Dynamics
  optimizer [1]. The optimization variable is regarded as a sample from the
  posterior under Stochastic Gradient Langevin Dynamics with noise rescaled in
  each dimension according to RMSProp [2].

  Note: If a prior is included in the loss, it should be scaled by
  `1/num_pseudo_batches`, where num_pseudo_batches is the number of minibatches
  in the data.  I.e., it should be divided by the `num_pseudo_batches` term
  described below.

  This function can update multiple chains in parallel. It assumes that all
  leftmost dimensions of `current_state` index independent chain states (and are
  therefore updated independently). The output of `target_log_prob_fn()` should
  sum log-probabilities across all event dimensions. Slices along the rightmost
  dimensions may have different target distributions; for example,
  `current_state[0, :]` could have a different target distribution from
  `current_state[1, :]`. This is up to `target_log_prob_fn()`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  [1]: "Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural
       Networks." Chunyuan Li, Changyou Chen, David Carlson, Lawrence Carin.
       ArXiv:1512.07666, 2015. https://arxiv.org/abs/1512.07666
  [2]: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

  Args:
    target_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the target distribution.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
    momentum: Tensor or List of Tensors, representing exponentially
      weighted moving average of each squared gradient with respect to a
      state. It is recommended to initialize it with tf.ones.
    learning_rate: Scalar `float`-like `Tensor`. The base learning rate for the
      optimizer. Must be tuned to the specific function being minimized.
    preconditioner_decay_rate: Scalar `float`-like `Tensor`. The exponential
      decay rate of the rescaling of the preconditioner (RMSprop). (This is
      "alpha" in [1]). Should be smaller than but nearly `1` to approximate
      sampling from the posterior. (Default: `0.95`)
    num_pseudo_batches: Scalar `int`-like `Tensor`. The effective number of
      minibatches in the data set.  Trades off noise and prior with the SGD
      likelihood term. Note: Assumes the loss is taken as the mean over a
      minibatch. Otherwise if the sum was taken, divide this number by the
      batch size.  (Default: `1`)
    burnin: Scalar `int`-like `Tensor`. The number of iterations to collect
      gradient statistics to update the preconditioner before starting to draw
      noisy samples. (Default: `25`)
    diagonal_bias: Scalar `float`-like `Tensor`. Term added to the diagonal of
      the preconditioner to prevent the preconditioner from degenerating.
      (Default: `1e-8`)
    seed: Python integer to seed the random number generator.
    current_target_log_prob: (Optional) `Tensor` representing the value of
      `target_log_prob_fn` at the `current_state`. The only reason to
      specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    current_grads_target_log_prob: (Optional) Python list of `Tensor`s
      representing gradient of `current_target_log_prob` at the `current_state`
      and wrt the `current_state`. Must have same shape as `current_state`. The
      only reason to specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "sgld_kernel").

  Returns:
    accepted_states: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state` but with a prepended `num_results`-size dimension.
    kernel_results: `collections.namedtuple` of internal calculations used to
      advance the chain.
  """
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import random_ops
  is_list_like = lambda x: isinstance(x, (tuple, list))
  maybe_list = lambda x: list(x) if is_list_like(x) else [x]
  states = maybe_list(current_state)
  momentums = maybe_list(momentum)
  with tf.name_scope(name, "sgld_kernel", states):
    with tf.name_scope("initialize"):
      if current_target_log_prob is None:
        current_target_log_prob = target_log_prob_fn(*states)
      if current_grads_target_log_prob is None:
        current_grads_target_log_prob = tf.gradients(current_target_log_prob, states)

    # TODO doesn't this scale the noise incorrectly by additional
    # learning_rate during the update? (same in sgld_optimizer)
    next_states = []
    momentums = []
    for state, mom, grad in zip(states, momentums, current_grads_target_log_prob):
      next_state = (
          state + learning_rate *
          _apply_noisy_update(mom, grad, learning_rate, diagonal_bias,
                              num_pseudo_batches, seed))
      momentum = (
          mom + (1.0 - preconditioner_decay_rate) * (tf.square(grad) - mom))
      next_states.append(next_state)
      momentums.append(momentum)

    maybe_flatten = lambda x: x if is_list_like(state) else x[0]
    next_state = maybe_flatten(next_states)
    momentum = maybe_flatten(momentums)
    return [
        next_state,
        momentum,
    ]


def _apply_noisy_update(mom, grad, learning_rate, diagonal_bias,
                        num_pseudo_batches, seed):
  # Compute and apply the gradient update following
  # preconditioned Langevin dynamics
  stddev = math_ops.cast(math_ops.rsqrt(learning_rate), grad.dtype)
  preconditioner = math_ops.rsqrt(
      mom + math_ops.cast(diagonal_bias, grad.dtype))
  return (
      0.5 * preconditioner * grad * math_ops.cast(num_pseudo_batches,
                                                  grad.dtype) +
      random_ops.random_normal(array_ops.shape(grad),
                               1.0,
                               dtype=grad.dtype,
                               seed=seed) *
      stddev * math_ops.sqrt(preconditioner))
