[![edward](../master/docs/images/edward_200.png?raw=true)](http://edwardlib.org)

[![Build Status](https://travis-ci.org/blei-lab/edward.svg?branch=master)](https://travis-ci.org/blei-lab/edward)
[![Coverage Status](https://coveralls.io/repos/github/blei-lab/edward/badge.svg?branch=master&cacheBuster=1)](https://coveralls.io/github/blei-lab/edward?branch=master)
[![Join the chat at https://gitter.im/blei-lab/edward](https://badges.gitter.im/blei-lab/edward.svg)](https://gitter.im/blei-lab/edward?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[Edward](http://edwardlib.org) is a Python library for probabilistic modeling,
inference, and criticism. It is a testbed for fast experimentation and research
with probabilistic models, ranging from classical hierarchical models on small
data sets to complex deep probabilistic models on large data sets. Edward fuses
three fields: Bayesian statistics and machine learning, deep learning, and
probabilistic programming.

It supports __modeling__ with

+ Directed graphical models
+ Neural networks (via libraries such as
    [`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers)
    and
    [Keras](http://keras.io))
+ Implicit generative models
+ Bayesian nonparametrics and probabilistic programs

It supports __inference__ with

+ Variational inference
  + Black box variational inference
  + Stochastic variational inference
  + Generative adversarial networks
  + Maximum a posteriori estimation
+ Monte Carlo
  + Gibbs sampling
  + Hamiltonian Monte Carlo
  + Stochastic gradient Langevin dynamics
+ Compositions of inference
  + Expectation-Maximization
  + Pseudo-marginal and ABC methods
  + Message passing algorithms

It supports __criticism__ of the model and inference with

+ Point-based evaluations
+ Posterior predictive checks

Edward is built on top of [TensorFlow](https://www.tensorflow.org).
It enables features such as computational graphs, distributed
training, CPU/GPU integration, automatic differentiation, and
visualization with TensorBoard.

## Resources

+ [Edward website](http://edwardlib.org)
+ [Edward Forum](http://discuss.edwardlib.org)
+ [Edward Gitter channel](http://gitter.im/blei-lab/edward)
+ [Edward releases](https://github.com/blei-lab/edward/releases)
+ [Edward papers, posters, and slides](https://github.com/edwardlib/papers)

See [Getting Started](http://edwardlib.org/getting-started) for how to install Edward.
