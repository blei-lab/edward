#### __Edward__ is a library for probabilistic modeling, inference, and criticism. 

Edward is a Python library that enables black box inference for models with discrete
and continuous latent variables, neural network parameterizations, and
infinite dimensional parameter spaces. Edward fuses three fields: Bayesian statistics and machine learning, deep learning,
and probabilistic programming.

It supports __modeling languages__ including

* [TensorFlow](https://www.tensorflow.org) (with neural network compositionality via [Keras](http://keras.io), [Pretty Tensor](https://github.com/google/prettytensor), or [TensorFlow-Slim](https://github.com/tensorflow/models/blob/master/inception/inception/slim/README.md))
* [Stan](http://mc-stan.org)
* [PyMC3](http://pymc-devs.github.io/pymc3/)
* Python through [NumPy/SciPy](http://scipy.org/)

It supports __inference__ via

* Variational inference
    * Black box variational inference
    * Stochastic variational inference
    * Variational auto-encoders
    * Inclusive KL divergence (KL(p || q))
* Marginal posterior optimization (empirical Bayes, marginal maximum likelihood)
* Maximum a posteriori estimation (penalized maximum likelihood, maximum likelihood)

It supports __criticism__ of the model and inference via

* Evaluations on both supervised and unsupervised tasks
* Posterior predictive checks

It also includes __features__ such as

* [TensorFlow](https://www.tensorflow.org) for backend computation, which includes automatic differentiation, GPU support, computational graphs, optimization, and TensorBoard
* A library for probability distributions in TensorFlow
* Documentation and tutorials
* Examples demonstrating state-of-the-art generative models and inference




#### Authors

Edward is led by [Dustin Tran](http://dustintran.com) with guidance by [David Blei](http://www.cs.columbia.edu/~blei/). It is under active development (by order of joining) by [Dustin Tran](http://dustintran.com), [David Blei](http://www.cs.columbia.edu/~blei/), [Alp Kucukelbir](http://www.proditus.com/), [Adji Dieng](http://stat.columbia.edu/~diengadji/), [Maja Rudolph](http://maja-rita-rudolph.com/), and [Dawen Liang](http://www.ee.columbia.edu/~dliang/). We welcome contributions by submitting issues, feature requests, or by solving any current issues!

We thank Rajesh Ranganath, Allison Chaney, Jaan Altosaar, and other members of the Blei Lab for their helpful feedback and advice.

#### Citation

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
