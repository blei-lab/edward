![edward](http://dustintran.com/img/blackbox_200.png)

__Edward__ is a Python library for probabilistic modeling, inference,
and criticism. It enables black box inference for models with discrete
and continuous latent variables, neural network parameterizations, and
infinite dimensional parameter spaces. Edward serves as a fusion of
three fields: Bayesian statistics and machine learning, deep learning,
and probabilistic programming.

It supports __modeling languages__ including

* [Stan](http://mc-stan.org)
* [TensorFlow](https://www.tensorflow.org) (with neural network compositionality via [Pretty Tensor](https://github.com/google/prettytensor) and [TensorFlow-Slim](https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md))
* original Python using [NumPy/SciPy](http://scipy.org/)

It supports __inference__ via

* Variational inference
  * Global divergence minimization
    * Black box variational inference
    * Stochastic variational inference
    * Variational auto-encoders
    * Inclusive KL divergence (KL(p || q))
  * Marginal posterior optimization (empirical Bayes, marginal maximum likelihood)
  * Maximum a posteriori (penalized maximum likelihood, maximum likelihood)

It also has __features__ including

* [TensorFlow](https://www.tensorflow.org) for backend computation, which includes automatic differentiation, GPU support, computational graphs, optimization, and TensorBoard
* A library for probability distributions in TensorFlow
* Documentation and tutorials
* Examples demonstrating state-of-the-art generative models and inference

## Getting Started

[You can find a tutorial here](https://github.com/blei-lab/edward/wiki/Tutorial) (TODO I think we should put a short tutorial here, or just demonstrate code snippets).
We highlight a few examples, more of which can be found in [`examples/`](examples/):

* [Convolutional variational auto-encoder](examples/convolutional_vae.py)

Read the documentation in the [Wiki](https://github.com/blei-lab/edward/wiki).

## Installation

To install from pip, run
```{bash}
pip install -e "git+https://github.com/blei-lab/edward.git#egg=edward"
```

## Authors

## Citation

We appreciate citations for Edward because it lets us find out how
people have been using the library and it motivates further work.

. 2016. Edward: A library for probabilistic modeling, inference, and criticism, Version 0.1. https://github.com/blei-lab/edward
```
@misc{
  author = {},
  title = {{Edward: A library for probabilistic modeling, inference, and criticism, Version 0.1}},
  year = {2016},
  url = {https://github.com/blei-lab/edward}
}
```
