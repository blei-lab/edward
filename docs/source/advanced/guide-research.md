## Guide for Research

This is a guide to how to use Edward for research. Following Box's loop, we divide research into three components: model, inference, and criticism.

As the library uses TensorFlow as a backend, here is a quick guide on [how to get started with it](tensorflow.md). You will most likely need to work directly in TensorFlow as you manipulate different objects and understand how certain behaviors of the new research works. Here is an [example](https://github.com/blei-lab/edward/blob/master/examples/normal_idiomatic_tf.py) with access to the TensorFlow session rather than hiding the TensorFlow internals with `inference.run`.

### Getting Started with TensorFlow

If you'd like to use this library for research, you have to learn TensorFlow. Trust me. It's worth the investment. As a heuristic, it takes roughly a day to have a good grasp of the essential mechanics behind the library.

Here's what I recommend for learning TensorFlow.

1. Read TensorFlow's [Getting Started](https://www.tensorflow.org/versions/r0.7/get_started/index.html). It tells you the essential objects that it works with.
2. Skim through the simple examples in these [TensorFlow tutorials](https://github.com/nlintz/TensorFlow-Tutorials). It gives you a big picture of the semantics and how the commands generally work with each other.
3. Skim through the example code in this library! If you’re familiar with the underlying math for variational inference, going through the code base here will also teach you the mapping from math to TensorFlow.


### Developing new probabilistic models

A probabilistic model is specified by a joint distribution `p(x,z)` of data `x` and latent variables `z`. All models in Edward are written as a class; to implement a new model, it can be written in any of the currently supported modeling languages: Stan, TensorFlow, and NumPy/SciPy.

To use Stan, simply write a Stan program in the form of a file or string. Then call it with `StanModel(file)` or `StanModel(model_code)`. Here is an example:
```{Python}
model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(1.0, 1.0);
      for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
"""
model = ed.StanModel(model_code=model_code)
```
Here is a [toy script](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_stan.py) that uses this model. Stan programs are convenient as [there are many online examples](https://github.com/stan-dev/example-models/wiki), although they are limited to probability models with differentiable latent variables and they can be quite slow to call in practice over TensorFlow.

To use TensorFlow, PyMC3, or NumPy/SciPy, write a class with the method `log_prob(xs, zs)`. This method takes as input a mini-batch of data `xs` and a mini-batch of the latent variables `zs`; the method outputs a vector of the joint density evaluations `[log p(xs, zs[0,:]), log p(xs, zs[1,:]), ...]` with size being the size of the latent variables' mini-batch. Here is an example:
```{Python}
class BetaBernoulli:
    """
    p(x, z) = Bernoulli(x | z) * Beta(z | 1, 1)
    """
    def __init__(self):
        self.num_vars = 1

    def log_prob(self, xs, zs):
        log_prior = beta.logpdf(zs, a=1.0, b=1.0)
        log_lik = tf.pack([tf.reduce_sum(bernoulli.logpmf(xs, z))
                           for z in tf.unpack(zs)])
        return log_lik + log_prior

model = BetaBernoulli()
```
Here is a [toy script](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_tf.py) that uses this model which is written in TensorFlow. Here is another [toy script](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_np.py) that uses the same model written in NumPy/SciPy and [another](https://github.com/blei-lab/edward/blob/master/examples/beta_bernoulli_pymc3.py) written in PyMC3.

For efficiency during future inferences or criticisms, we recommend using the modeling language which contains the most structure about the model; this enables the inference algorithms to automatically take advantage of any available structure if they are implemented to do so. TensorFlow will be most efficient as Edward uses it as the backend for computation.

### Developing new inference algorithms

An inference algorithm calculates the posterior for a particular model and data set; it is the distribution of the latent variables given data, `p(z | x)`, and is used in all downstream analyses such as prediction. With Edward, you can develop new black box inference algorithms and also develop custom inference algorithms which are tailored to a particular model or restricted class of models.

There is a base `Inference` class, from which all inference methods are based on. We categorize inference under two paradigms:

* `VariationalInference`
* `MonteCarlo`

(or more plainly, optimization and sampling). These inherit from `Inference` and each have their own default methods. See the file [`inferences.py`](https://github.com/blei-lab/edward/blob/master/edward/inferences.py).

Consider developing a variational inference algorithm.
The main method in `VariationalInference` is `run()`, which is a simple wrapper that first runs `initialize()` and then in a loop runs `update()` and `print_progress()`. To develop a new variational inference algorithm, inherit from `VariationalInference` and write a new method for `build_loss()`: this returns an object that TensorFlow will automatically differentiate during optimization. The other methods have defaults which you can update as necessary. The [inclusive KL divergence algorithm in `inferences.py`](https://github.com/blei-lab/edward/blob/master/edward/inferences.py) is a useful example. It writes `build_loss()` so that automatic diferentiation of its return object is a tractable gradient that minimizes KL(p||q). It also modifies `initialize()` and `update()`.

Consider developing a Monte Carlo algorithm. Inherit from `MonteCarlo`.[Documentation is in progress.]

Note that you can build model-specific inference algorithms and inference algorithms that are tailored to a smaller class than the general class available here. There's nothing preventing you to do so, and the general organizational paradigm and low-level functions are still useful in such a case. You can write a class that for example inherits from `Inference` directly or inherits to carry both optimization and sampling methods.

### Developing new criticism techniques

[Documentation is in progress.]


### Developing new building blocks

Here is a list of advanced features currently supported.

+ Score function gradient
+ Reparameterization gradient
+ Variance reduction techniques
    + Analytic KL's and entropy decompositions
+ Variational models
    + Mean-field
    + Global parameterizations (inference networks, recognition networks, inverse mappings)
