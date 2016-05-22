[![edward](http://dustintran.com/img/blackbox_200.png)](http://www.erikolofsen.com/blackbox.html)

__Edward__ is a Python library for probabilistic modeling, inference,
and criticism. It enables black box inference for models with discrete
and continuous latent variables, neural network parameterizations, and
infinite dimensional parameter spaces. Edward serves as a fusion of
three fields: Bayesian statistics and machine learning, deep learning,
and probabilistic programming.

It supports __modeling languages__ including

* [TensorFlow](https://www.tensorflow.org) (with neural network compositionality via [Pretty Tensor](https://github.com/google/prettytensor) and [TensorFlow-Slim](https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md))
* [Stan](http://mc-stan.org)
* [PyMC3](http://pymc-devs.github.io/pymc3/)
* original Python using [NumPy/SciPy](http://scipy.org/)

It supports __inference__ via

* Variational inference
  * Divergence minimization
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

[You can find a tutorial here](https://github.com/blei-lab/edward/wiki/Tutorial)
for getting started with Edward, as well as a
[tutorial here](https://github.com/blei-lab/edward/wiki/Tutorial-for-Research)
for how to use it for research. We highlight a few examples, more of
which can be found in [`examples/`](examples/):

* [Bayesian linear regression](examples/bayesian_linear_regression.py)
* [Hierarchical logistic regression](examples/hierarchical_logistic_regression.py)
* [Mixture model of Gaussians](examples/mixture_gaussian.py)
* [Gaussian process classification](examples/gp_classification.py)
* [Bayesian neural network](examples/bayesian_nn.py)
* [Mixture density network](examples/mixture_density_network.py)
* [Variational auto-encoder](examples/convolutional_vae.py)

Read the documentation in the [Wiki](https://github.com/blei-lab/edward/wiki).

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

Dustin Tran, David M. Blei, Alp Kucukelbir, Adji Dieng, Maja Rudolph, and Dawen Liang. 2016. Edward: A library for probabilistic modeling, inference, and criticism. https://github.com/blei-lab/edward
```
@misc{tran2016edward,
  author = {Dustin Tran and David M. Blei and Alp Kucukelbir and Adji Dieng and Maja Rudolph and Dawen Liang},
  title = {{Edward: A library for probabilistic modeling, inference, and criticism}},
  year = {2016},
  url = {https://github.com/blei-lab/edward}
}
```
