[![edward](http://dustintran.com/img/blackbox_200.png)](http://www.erikolofsen.com/blackbox.html)

[![Build Status](https://travis-ci.org/blei-lab/edward.svg?branch=master)](https://travis-ci.org/blei-lab/edward)
[![Coverage Status](https://coveralls.io/repos/github/blei-lab/edward/badge.svg?branch=master)](https://coveralls.io/github/blei-lab/edward?branch=master)

__Edward__ is a Python library for probabilistic modeling, inference,
and criticism. It is a testbed for fast experimentation and research
with probabilistic models, ranging from classical hierarchical
models on small data sets to complex deep probabilistic models on
large data sets. Edward fuses three fields: Bayesian statistics and
machine learning, deep learning, and probabilistic programming.

It supports __modeling languages__ including

+ [TensorFlow](https://www.tensorflow.org) (with neural networks via [Keras](http://keras.io), [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim), or [Pretty Tensor](https://github.com/google/prettytensor))
+ [Stan](http://mc-stan.org)
+ [PyMC3](http://pymc-devs.github.io/pymc3/)
+ Python, through [NumPy/SciPy](http://scipy.org/)

It supports __inference__ via

+ Variational inference
    + Black box variational inference
    + Stochastic variational inference
    + Variational auto-encoders
    + Inclusive KL divergence (KL(p || q))
    + Laplace approximation
+ Marginal posterior optimization (empirical Bayes, marginal maximum likelihood)
+ Maximum a posteriori (penalized maximum likelihood, maximum likelihood)

It supports __criticism__ of the model and inference via

+ Evaluations on both supervised and unsupervised tasks
+ Posterior predictive checks

It also has __features__ including

+ [TensorFlow](https://www.tensorflow.org) for backend computation, which includes computational graphs, distributed training, CPU/GPU integration, automatic differentiation, and TensorBoard
+ A library for probability distributions in TensorFlow
+ Documentation and tutorials
+ Examples demonstrating state-of-the-art generative models and inference

## Getting Started

[You can find a tutorial here](http://edwardlib.org/getting-started/)
for getting started with Edward. We highlight a few examples, more of
which can be found in [`examples/`](examples/):

+ [Bayesian linear regression](examples/bayesian_linear_regression.py)
+ [Hierarchical logistic regression](examples/hierarchical_logistic_regression.py)
+ [Mixture model of Gaussians](examples/mixture_gaussian.py)
+ [Gaussian process classification](examples/gp_classification.py)
+ [Bayesian neural network](examples/bayesian_nn.py)
+ [Mixture density network](examples/mixture_density_network.py)
+ [Variational auto-encoder](examples/convolutional_vae.py)
+ [Importance-weighted variational inference](examples/iwvi.py)

Read the documentation on the [website](http://edwardlib.org).

## Installation

To install the latest stable version, run
```{bash}
pip install edward
```
To install the latest development version, run
```{bash}
pip install -e "git+https://github.com/blei-lab/edward.git#egg=edward"
```

## Authors

Edward is led by [Dustin Tran](http://dustintran.com) with guidance by [David Blei](http://www.cs.columbia.edu/~blei/). It is under active development (by order of joining) by [Dustin Tran](http://dustintran.com), [David Blei](http://www.cs.columbia.edu/~blei/), [Alp Kucukelbir](http://www.proditus.com/), [Adji Dieng](http://stat.columbia.edu/~diengadji/), [Maja Rudolph](http://maja-rita-rudolph.com/), and [Dawen Liang](http://www.ee.columbia.edu/~dliang/). We welcome contributions by submitting issues, feature requests, or by solving any current issues!

We thank Rajesh Ranganath, Allison Chaney, Jaan Altosaar, and other members of the Blei Lab for their helpful feedback and advice.

## Citation

We appreciate citations for Edward because it lets us find out how
people have been using the library and it motivates further work.

Dustin Tran, David M. Blei, Alp Kucukelbir, Adji Dieng, Maja Rudolph, and Dawen Liang. 2016. Edward: A library for probabilistic modeling, inference, and criticism. http://edwardlib.org
```
@misc{tran2016edward,
  author = {Dustin Tran and David M. Blei and Alp Kucukelbir and Adji Dieng and Maja Rudolph and Dawen Liang},
  title = {{Edward: A library for probabilistic modeling, inference, and criticism}},
  year = {2016},
  url = {http://edwardlib.org}
}
```
