"""Programmable docstrings.

The args below represent a global vocabulary of arguments shared
across at least two inference algorithms. They are sorted
alphabetically. They are also written with newlines at the end such
that they can be easily added together. After composing args
docstrings, remove the last newline.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
import sys


def set_doc(**kwargs):
  """Decorator to programmatically set the docstring."""
  def _update(cls_or_fn):
    doc = trim(cls_or_fn.__doc__)
    for k, v in six.iteritems(kwargs):
      # Capture each @{k} reference to replace with v.
      # We wrap the replacement in a function so no backslash escapes
      # are processed.
      pattern = r'@\{' + str(k) + r'\}'
      doc = re.sub(pattern, lambda match: v, doc)
    cls_or_fn.__doc__ = doc
    return cls_or_fn
  return _update


def trim(docstring):
  """Trims docstring indentation. Taken from PEP 257 docs."""
  if not docstring:
    return ''
  # Convert tabs to spaces (following the normal Python rules)
  # and split into a list of lines:
  lines = docstring.expandtabs().splitlines()
  # Determine minimum indentation (first line doesn't count):
  indent = sys.maxint
  for line in lines[1:]:
    stripped = line.lstrip()
    if stripped:
      indent = min(indent, len(line) - len(stripped))
  # Remove indentation (first line is special):
  trimmed = [lines[0].strip()]
  if indent < sys.maxint:
    for line in lines[1:]:
      trimmed.append(line[indent:].rstrip())
  # Strip off trailing and leading blank lines:
  while trimmed and not trimmed[-1]:
    trimmed.pop()
  while trimmed and not trimmed[0]:
    trimmed.pop(0)
  # Return a single string:
  return '\n'.join(trimmed)


arg_align_data = """
  align_data: function of string, aligning `model` observed
    variables with data. It takes a model variable's name as input
    and returns an integer, indexing `args`, or key, indexing
    `kwargs`. Other inputs must return None.
"""[1:]
arg_align_latent = """
  align_latent: function of string, aligning `model` latent
    variables with `variational`. It takes a model variable's name
    as input and returns a string, indexing `variational`'s trace.
    Other inputs must return None.
"""[1:]
arg_align_latent_monte_carlo = """
  align_latent: function of string, aligning `model` latent
    variables with posterior trace. It takes a model variable's name
    as input and returns a string. The return output determines the
    name of the returned dictionary of states' keys. If None,
    will not perform inference over them.
"""[1:]
arg_args_kwargs = """
  args, kwargs: data inputs. `kwargs`' keys are directly the argument
    keys in `model` (and if present, `variational`). Data inputs are
    passed at compile-time in TF's Graph mode or runtime in TF's Eager
    mode.
"""[1:]
arg_auto_transform = """
  auto_transform:
"""[1:]
arg_collections = """
  collections:
"""[1:]
arg_discriminator = """
  discriminator: function.
    Function (with parameters) to discriminate samples. It should
    output logit probabilities (real-valued) and not probabilities
    in $[0, 1]$.
"""[1:]
arg_current_grads_target_log_prob = """
  current_grads_target_log_prob:
"""[1:]
arg_kl_scaling = """
  kl_scaling: function of string, aligning `model` latent
    variables with KL scale factors. This provides option to scale
    terms when using ELBO with KL divergence. If the KL divergence
    terms are

    $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
          \log q(z\mid x, \lambda) - \log p(z)],$

    then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
    where $\\alpha_p$ is a tensor. Its shape must be broadcastable;
    it is multiplied element-wise to the batchwise KL terms.
"""[1:]
arg_model = """
  model: function whose inputs are a subset of `args` (e.g., for
    discriminative). Output is not used.
    TODO auto_transform docstring
    Collection of random variables to perform inference on.
    If list, each random variable will be implictly optimized using
    a `Normal` random variable that is defined internally with a
    free parameter per location and scale and is initialized using
    standard normal draws. The random variables to approximate must
    be continuous.
    TODO note above only applicable to variational(?) inferences
"""[1:]
arg_n_samples = """
  n_samples: int.
    Number of samples from variational model for calculating
    stochastic gradients.
"""[1:]
arg_scale = """
  scale: function of string, aligning `model` observed
    variables with scale factors. It takes a model variable's name
    as input and returns a scale factor; else 1.0. The scale
    factor's shape must be broadcastable; it is multiplied
    element-wise to the random variable. For example, this is useful
    for mini-batch scaling when inferring global variables, or
    applying masks on a random variable.
"""[1:]
arg_current_state = """
  current_state: Tensor or list of Tensors. Each element is a
    posterior variable whose name is its current state. If the model
    encounters a latent variable not aligned with a key in `states`,
    its state is a draw from the distribution. Default is None
    (equivalent to empty dict).
"""[1:]
arg_step_size = """
  step_size: float.
    Step size of numerical integrator. The implementation may be
    extended in the future to enable a step size per random variable
    (`step_size` would be a callable).
"""[1:]
arg_current_target_log_prob = """
  current_target_log_prob:
"""[1:]
arg_variational = """
  variational: function whose inputs are a subset of `args` (e.g.,
    for amortized). Output is not used.
"""[1:]
notes_conditional_inference = """
In conditional inference, we infer $z$ in $p(z, \\beta
\mid x)$ while fixing inference over $\\beta$ using another
distribution $q(\\beta)$. During calculations, this function uses an
estimate of the marginal density,

$\log p(x, z) = \log \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
              \\approx \log p(x, z, \\beta^*)$

leveraging a single Monte Carlo sample, where $\\beta^* \sim
q(\\beta)$. This is unbiased (and therefore asymptotically exact as a
pseudo-marginal method) if $q(\\beta) = p(\\beta \mid x)$.
"""[1:-1]
notes_conditional_inference_samples = """
In conditional inference, we infer $z$ in $p(z, \\beta
\mid x)$ while fixing inference over $\\beta$ using another
distribution $q(\\beta)$. During gradient calculation, instead
of using the model's density

$\log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),$

for each sample $s=1,\ldots,S$, this function uses

$\log p(x, z^{(s)}, \\beta^{(s)}),$

where $z^{(s)} \sim q(z; \lambda)$ and $\\beta^{(s)}
\sim q(\\beta)$.
"""[1:-1]
notes_discriminator_scope = """
In building the computation graph for inference, the
discriminator's parameters can be accessed with the variable scope
"Disc".
"""[1:-1]
notes_mcmc_programs = """
Probabilistic programs may have random variables which vary across
executions. At each iteration, the MCMC algorithm transitions across
the (finite) list of latent variables seen during one execution of
the model. The previous state is read from `states`: if the
execution encounters a latent variable not existing in `states`, the
previous state is a draw from the prior.

We recommend updating `states` with the sampler's output after each
iteration. For example, in Eager mode:
```python
states = {}
for _ in range(10000):
  new_states, ... = mcmc(..., states=states, ...)
  states.update(new_states)
```
This caches previous states within the `states` dictionary. States
are only updated when the associated latent variable is seen again
in the model's execution. As long as every latent variable of
interest appears in the execution with non-zero probability, the
distribution of each state is guaranteed to converge to the target
distribution.

This idea can be seen as a joint version of single-site
Metropolis-Hastings [@wingate2011lightweight], but note it does not
rerun any part of the program. In fact, the newly transitioned states
given old states may not actually be a valid output of the program.
For example, consider
```python
def model():
  x = Bernoulli(probs=0.5)
  if tf.cast(x, tf.bool):
    y = Normal(0.0, 1.0)
  else:
    y = Gamma(1.0, 1.0)
  return x, y
```
Given a previous state from (Bernoulli, Normal), the proposal might
generate (0, -0.3), which is not in the program's support.
"""[1:-1]
notes_model_parameters = """
The function also enables optimizing model parameters $p(z \mid x;
\\theta)$. It does this by variational EM, maximizing

$\mathbb{E}_{q(z; \lambda)} [ \log p(x, z; \\theta) ]$

with respect to $\\theta$.
"""[1:-1]
notes_regularization_losses = """
The objective function also adds to itself a summation over all
tensors in the `REGULARIZATION_LOSSES` collection.
"""
return_loss = """
  Scalar tf.Tensor representing the loss. Its automatic
  differentiation is the gradient to follow for optimization.
"""[1:-1]
return_loss_loss_d = """
  Pair of scalar tf.Tensors, representing the generative loss and
  discriminative loss respectively.
"""[1:-1]
return_loss_surrogate_loss = """
  Pair of scalar tf.Tensors, representing the loss and surrogate loss
  respectively. The surrogate loss' automatic differentiation is the
  gradient to follow for optimization.
"""[1:-1]
return_samples = """
  Dict of tf.Tensor. The keys are according to the return values of
  `align_latent`. The associated values are the transitioned states
  from the Markov chain.
"""[1:-1]
return_surrogate_loss = """
  Scalar tf.Tensor representing the surrogate loss. Its automatic
  differentiation is the gradient to follow for optimization.
"""[1:-1]
