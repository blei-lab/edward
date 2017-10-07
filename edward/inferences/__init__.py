"""Algorithms for inferring parameters or latent variables.

We describe how to perform inference in probabilistic models.
For background, see the
[Inference tutorial](/tutorials/inference).

Suppose we have a model $p(\mathbf{x}, \mathbf{z}, \\beta)$ of data
$\mathbf{x}_{\\text{train}}$ with latent variables $(\mathbf{z}, \\beta)$.
Consider the posterior inference problem,

$$
q(\mathbf{z}, \\beta)\\approx p(\mathbf{z}, \\beta\mid \mathbf{x}_{\\text{train}}),
$$

in which the task is to approximate the posterior
$p(\mathbf{z}, \\beta\mid \mathbf{x}_{\\text{train}})$
using a family of distributions, $q(\mathbf{z},\\beta; \lambda)$,
indexed by parameters $\lambda$.

In Edward, let `z` and `beta` be latent variables in the model,
where we observe the random variable `x` with
data `x_train`.
Let `qz` and `qbeta` be random variables defined to
approximate the posterior.
We write this problem as follows:

```python
inference = ed.Inference({z: qz, beta: qbeta}, {x: x_train})
```

`Inference` is an abstract class which takes two inputs.  The
first is a collection of latent random variables `beta` and
`z`, along with "posterior variables" `qbeta` and
`qz`, which are associated to their respective latent
variables.  The second is a collection of observed random variables
`x`, which is associated to the data `x_train`.

Inference adjusts parameters of the distribution of `qbeta`
and `qz` to be close to the
posterior $p(\mathbf{z}, \\beta\,|\,\mathbf{x}_{\\text{train}})$.

Running inference is as simple as running one method.

```python
inference = ed.Inference({z: qz, beta: qbeta}, {x: x_train})
inference.run()
```

Inference also supports fine control of the training procedure.

```python
inference = ed.Inference({z: qz, beta: qbeta}, {x: x_train})
inference.initialize()

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

inference.finalize()
```

`initialize()` builds the algorithm's update rules
(computational graph) for $\lambda$;
`tf.global_variables_initializer().run()` initializes $\lambda$
(TensorFlow variables in the graph);
`update()` runs the graph once to update
$\lambda$, which is called in a loop until convergence;
`finalize()` runs any computation as the algorithm
terminates.

The `run()` method is a simple wrapper for this procedure.

### Other Settings

We highlight other settings during inference.

__Model parameters__.
Model parameters are parameters in a model that we will always compute
point estimates for and not be uncertain about.
They are defined with `tf.Variable`s, where the inference
problem is

$$
\hat{\\theta} \leftarrow^{\\text{optimize}}
p(\mathbf{x}_{\\text{train}}; \\theta)
$$

```python
from edward.models import Normal

theta = tf.Variable(0.0)
x = Normal(loc=tf.ones(10) * theta, scale=1.0)

inference = ed.Inference({}, {x: x_train})
```

Only a subset of inference algorithms support estimation of model
parameters.
(Note also that this inference example does not have any latent
variables. It is only about estimating `theta` given that we
observe $\mathbf{x} = \mathbf{x}_{\\text{train}}$. We can add them so
that inference is both posterior inference and parameter estimation.)

For example, model parameters are useful when applying neural networks
from high-level libraries such as Keras and TensorFlow Slim. See
the [model compositionality](/tutorials/model-compositionality) page
for more details.

__Conditional inference__.
In conditional inference, only a subset of the posterior is inferred
while the rest are fixed using other inferences. The inference
problem is

$$
q(\mathbf{z}\mid\\beta)q(\\beta)\\approx
p(\mathbf{z}, \\beta\mid\mathbf{x}_{\\text{train}})
$$

where parameters in $q(\mathbf{z}\mid\\beta)$ are estimated and
$q(\\beta)$ is fixed.
In Edward, we enable conditioning by binding random variables to other
random variables in `data`.
```python
inference = ed.Inference({z: qz}, {x: x_train, beta: qbeta})
```

In the [compositionality tutorial](/tutorials/inference-compositionality),
we describe how to construct inference by composing
many conditional inference algorithms.

__Implicit prior samples__.
Latent variables can be defined in the model without any posterior
inference over them. They are implicitly marginalized out with a
single sample. The inference problem is
$$$
q(\\beta)\\approx
p(\\beta\mid\mathbf{x}_{\\text{train}}, \mathbf{z}^*)
$$
where $\mathbf{z}^*\sim p(\mathbf{z}\mid\\beta)$ is a prior sample.

```python
inference = ed.Inference({beta: qbeta}, {x: x_train})
```

For example, implicit prior samples are useful for generative adversarial
networks. Their inference problem does not require any inference over
the latent variables; it uses samples from the prior.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward.inferences.bigan_inference import *
from edward.inferences.conjugacy import *
from edward.inferences.gan_inference import *
from edward.inferences.gibbs import *
from edward.inferences.hmc import *
from edward.inferences.implicit_klqp import *
from edward.inferences.inference import *
from edward.inferences.klpq import *
from edward.inferences.klqp import *
from edward.inferences.laplace import *
from edward.inferences.map import *
from edward.inferences.metropolis_hastings import *
from edward.inferences.monte_carlo import *
from edward.inferences.sgld import *
from edward.inferences.sghmc import *
from edward.inferences.variational_inference import *
from edward.inferences.wake_sleep import *
from edward.inferences.wgan_inference import *

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    'BiGANInference',
    'complete_conditional',
    'GANInference',
    'Gibbs',
    'HMC',
    'ImplicitKLqp',
    'Inference',
    'KLpq',
    'KLqp',
    'ReparameterizationKLqp',
    'ReparameterizationKLKLqp',
    'ReparameterizationEntropyKLqp',
    'ScoreKLqp',
    'ScoreKLKLqp',
    'ScoreEntropyKLqp',
    'ScoreRBKLqp',
    'Laplace',
    'MAP',
    'MetropolisHastings',
    'MonteCarlo',
    'SGLD',
    'SGHMC',
    'VariationalInference',
    'WakeSleep',
    'WGANInference',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
