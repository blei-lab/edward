![blackbox](http://dustintran.com/img/blackbox_200.png)

__blackbox__ is a probabilistic programming tool written in Python. It enables
black box inference for probabilistic models, including those with
discrete and continuous latent variables, neural network
parameterizations, and infinite dimensional parameter spaces. It is a
fusion of three fields: Bayesian statistics and machine learning, deep
learning, and probabilistic programming.

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

It also has __features__ including

* [TensorFlow](https://www.tensorflow.org) for backend computation, which includes automatic differentiation, GPU support, computational graphs, optimization, and TensorBoard
* A library for probability distributions in TensorFlow
* Documentation and tutorials
* Examples demonstrating state-of-the-art generative models and inference

[A list of advanced features is written here](https://github.com/Blei-Lab/blackbox/wiki), along with references.

## Getting Started

[You can find a tutorial here](https://github.com/Blei-Lab/blackbox/wiki/Tutorial).
Here is an example of variational inference on a Beta-Binomial model written in Stan:
```{Python}
import blackbox as bb

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
bb.set_seed(42)
model = bb.StanModel(model_code=model_code)
variational = bb.MFBeta(1)
data = bb.Data(dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]))

# Mean-field variational inference
inference = bb.MFVI(model, variational, data)
inference.run()
```
We also have a
[TensorFlow version](examples/beta_bernoulli_tf.py) and
[NumPy/SciPy version](examples/beta_bernoulli_np.py) of the example.
More examples are located in [`examples/`](examples/). We highlight a
few:

* [TODO]()

## Documentation

All documentation is available in the [Wiki](https://github.com/Blei-Lab/blackbox/wiki).

## Installation

To install from pip, run
```{bash}
pip install -e "git+https://github.com/blei-lab/blackbox.git#egg=blackbox"
```

## Authors

## Citation

We appreciate citations for blackbox because it lets us find out how
people have been using the library and it motivates further work.

. 2016. blackbox: Black box inference for probabilistic models, Version 0.1.   https://github.com/Blei-Lab/blackbox
```
@misc{
  author = {},
  title = {{blackbox: Black box inference for probabilistic models, Version 0.1}},
  year = {2016},
  url = {https://github.com/Blei-Lab/blackbox}
}
```
