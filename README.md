[![edward](../master/docs/images/edward_200.png?raw=true)](http://edwardlib.org)

[![Build Status](https://travis-ci.org/blei-lab/edward.svg?branch=master)](https://travis-ci.org/blei-lab/edward)
[![Coverage Status](https://coveralls.io/repos/github/blei-lab/edward/badge.svg?branch=master)](https://coveralls.io/github/blei-lab/edward?branch=master)

[Edward](http://edwardlib.org) is a Python library for probabilistic modeling,
inference, and criticism. It is a testbed for fast experimentation and research
with probabilistic models, ranging from classical hierarchical models on small
data sets to complex deep probabilistic models on large data sets. Edward fuses
three fields: Bayesian statistics and machine learning, deep learning, and
probabilistic programming.

It supports __modeling languages__ including

+ [TensorFlow](https://www.tensorflow.org) (with neural networks via
    [Keras](http://keras.io), [TensorFlow
    Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim),
    or [Pretty Tensor](https://github.com/google/prettytensor))
+ [Stan](http://mc-stan.org)
+ [PyMC3](http://pymc-devs.github.io/pymc3/)
+ Python, through [NumPy/SciPy](http://scipy.org/)

It supports __inference__ via

+ Variational inference
  + Black box variational inference
  + Stochastic variational inference
  + Variational auto-encoders
  + Inclusive KL divergence: KL(p||q)
+ Marginal posterior optimization (empirical Bayes, marginal
    maximum likelihood)
  + Variational EM
+ Maximum a posteriori estimation (penalized maximum likelihood,
    maximum likelihood)
  + Laplace approximation

It supports __criticism__ of the model and inference via

+ Point-based evaluations
+ Posterior predictive checks

Edward is built on top of [TensorFlow](https://www.tensorflow.org),
enabling features such as computational graphs, distributed training,
CPU/GPU integration, automatic differentiation, and visualization with
TensorBoard.

## Resources

+ [Edward website](http://edwardlib.org)
