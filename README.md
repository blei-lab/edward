![blackbox](http://dustintran.com/img/blackbox_200.png)

__blackbox__ is a tool for performing Bayesian inference over a large
class of models, including both discrete and continuous latent
variables. It enables rapid prototyping of models.

Example (more found in [`examples/`](examples/)):
```{Python}
import blackbox as bb

# demo here
```

## Features

(in progress)
* [TensorFlow](https://www.tensorflow.org)
* variational models ([Ranganath et al., 2015](http://arxiv.org/abs/1511.02386); [Tran et al., 2016](http://arxiv.org/abs/1511.06499))
* $f$-divergences
* approximate predictive checks
* inference networks ([Kingma and Welling, 2014](http://arxiv.org/abs/1312.6114); [Rezende et al., 2014](http://arxiv.org/abs/1401.4082))
* stochastic dynamics ([Salimans et al., 2015](http://arxiv.org/abs/1410.6460))
* Bayesian nonparametrics ([Kurihara et al., 2006](http://papers.nips.cc/paper/3025-accelerated-variational-dirichlet-process-mixtures.pdf))
* multicanonical ([Mandt et al., 2016](http://arxiv.org/abs/1411.1810))
* streaming ([McInerney et al., 2015](http://arxiv.org/abs/1507.05253))

other keywords: "black box variational inference", "automatic
differentiation variational inference", "variational autoencoders",
"deep learning"

## Installation

To install from pip, run
```{bash}
pip install -e "git+https://github.com/blei-lab/blackbox.git#egg=blackbox"
```

## Authors

## Citation

```
@misc{
  author = {},
  title = {},
  url={https://github.com/Blei-Lab/blackbox},
  year = {2016}
}
```
