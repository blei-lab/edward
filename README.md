[![edward](../master/docs/images/edward_200.png?raw=true)](http://edwardlib.org)

[![Build Status](https://travis-ci.org/blei-lab/edward.svg?branch=master)](https://travis-ci.org/blei-lab/edward)
[![Coverage Status](https://coveralls.io/repos/github/blei-lab/edward/badge.svg?branch=master)](https://coveralls.io/github/blei-lab/edward?branch=master)

[Edward](http://edwardlib.org) is a Python library for probabilistic modeling,
inference, and criticism. It is a testbed for fast experimentation and research
with probabilistic models, ranging from classical hierarchical models on small
data sets to complex deep probabilistic models on large data sets. Edward fuses
three fields: Bayesian statistics and machine learning, deep learning, and
probabilistic programming.

It supports __modeling__ with

+ Directed graphical models
+ Neural networks (via libraries such as
    [Keras](http://keras.io) and [TensorFlow
    Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim))
+ Conditionally specified undirected models
+ Bayesian nonparametrics and probabilistic programs

It supports __inference__ with

+ Variational inference
  + Black box variational inference
  + Stochastic variational inference
  + Variational auto-encoders
  + Inclusive KL divergence: KL(p||q)
+ Monte Carlo
  + Metropolis-Hastings
+ Marginal optimization (empirical Bayes, marginal maximum likelihood)
  + Variational EM
+ Maximum a posteriori estimation (penalized maximum likelihood,
    maximum likelihood)
  + Laplace approximation

It supports __criticism__ of the model and inference with

+ Point-based evaluations
+ Posterior predictive checks

Edward is built on top of [TensorFlow](https://www.tensorflow.org).
It enables features such as computational graphs, distributed
training, CPU/GPU integration, automatic differentiation, and
visualization with TensorBoard.

## Resources

+ [Edward website](http://edwardlib.org)
