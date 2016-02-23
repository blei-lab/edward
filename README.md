![blackbox](http://dustintran.com/img/blackbox_200.png)

__blackbox__ is a probabilistic programming tool implemented in Python
with [TensorFlow](https://www.tensorflow.org) as a backend. It enables
automatic Bayesian inference over a large class of models, including
both discrete and continuous latent variables.
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
data = dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

bb.set_seed(42)
model = bb.StanModel(model_code=model_code, data=data)
q = bb.MFBeta(model.num_vars)

# Mean-field variational inference
inference = bb.MFVI(model, q)
inference.run()
```
The equivalent example is also written in
[TensorFlow](examples/beta_bernoulli_tf.py) and
[NumPy/SciPy](examples/beta_bernoulli_np.py).
More examples are located in [`examples/`](examples/). We highlight a
few:

* [TODO]()

(This list will be removed; each item will be a specific example in
the list of highlighted examples above.)

* variational models ([Ranganath et al., 2015](http://arxiv.org/abs/1511.02386); [Tran et al., 2016](http://arxiv.org/abs/1511.06499))
* $f$-divergences
* approximate predictive checks
* inference networks ([Kingma and Welling, 2014](http://arxiv.org/abs/1312.6114); [Rezende et al., 2014](http://arxiv.org/abs/1401.4082))
* stochastic dynamics ([Salimans et al., 2015](http://arxiv.org/abs/1410.6460))
* Bayesian nonparametrics ([Kurihara et al., 2006](http://papers.nips.cc/paper/3025-accelerated-variational-dirichlet-process-mixtures.pdf))
* multicanonical ([Mandt et al., 2016](http://arxiv.org/abs/1411.1810))
* streaming ([McInerney et al., 2015](http://arxiv.org/abs/1507.05253))

## Installation

To install from pip, run
```{bash}
pip install -e "git+https://github.com/blei-lab/blackbox.git#egg=blackbox"
```

## Authors

## Citation

```
@misc{
  title = {{blackbox}: {B}lack box inference for probabilistic models},
  author = {},
  note = {Python package version 0.1},
  url = {https://github.com/Blei-Lab/blackbox},
  year = {2016}
}
```
