![blackbox](http://dustintran.com/img/blackbox_200.png)

__blackbox__ is a probabilistic programming tool implemented in Python
with [TensorFlow](https://www.tensorflow.org) as a backend. It enables
automatic Bayesian inference over a large class of models, including
those with both discrete and continuous latent variables.
Three modeling languages are supported:
[Stan](http://mc-stan.org), [TensorFlow](https://www.tensorflow.org),
and original Python using [NumPy/SciPy](http://scipy.org).

Example of inference on a Beta-Binomial model written in Stan:
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
The equivalent example is also written in
[TensorFlow](examples/beta_bernoulli_tf.py) and
[NumPy/SciPy](examples/beta_bernoulli_np.py).
More examples are located in [`examples/`](examples/). We highlight a
few:

* [TODO]()

[A comprehensive list of features is written here](https://github.com/Blei-Lab/blackbox/wiki), along with references.

## Documentation

[A tutorial is available here](https://github.com/Blei-Lab/blackbox/wiki/Tutorial).
More generally, all documentation is available in the [Wiki](https://github.com/Blei-Lab/blackbox/wiki).

## Installation

To install from pip, run
```{bash}
pip install -e "git+https://github.com/blei-lab/blackbox.git#egg=blackbox"
```

## Authors

## Citation

```
@misc{
  title = {{blackbox: Black box inference for probabilistic models}},
  author = {},
  note = {Python package version 0.1},
  url = {https://github.com/Blei-Lab/blackbox},
  year = {2016}
}
```
