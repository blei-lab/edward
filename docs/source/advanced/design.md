## Design and Philosophy

The __Edward__ library serves two purposes:

1. to serve as a foundation for research;

2. to provide an open-source, unified library for inference and criticism, with modeling tools at our disposal.

As a research tool, the code base is easily interchangeable so that we can play with extensions for further avenues of investigation, without having to re-implement everything from scratch. For example, for testing new inference algorithms, the model experiments only need to be written once, and we can simply swap the inference object with the newly proposed algorithm. For certain inference algorithms, this corresponds to inheriting from a current inference class and writing a new function for computing stochastic gradients. The same idea applies at a lower level: we can leverage the optimizers and visualization tools already available.

As a practical tool, Edward has many desirable features such as computational graphs for automatic differentiation, GPU support and distributed implementations, support for a variety of modeling languages. It also has the fundamentals: linear model examples, data subsampling for scaling to massive data sets, mean-field variational inference, and so on.

### Related Software

There are several notable themes in Edward.

__Probabilistic programming.__
There has been incredible progress recently on developing generic languages for specifying probability models, or probabilistic programs. Among these include [Venture](http://probcomp.csail.mit.edu/venture/), [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/literature/index.html), [Figaro](https://www.cra.com/work/case-studies/figaro), [Stan](http://mc-stan.org), and [WebPPL](http://dippl.org/chapters/02-webppl.html). We do not work on modeling languages; we work on general-purpose algorithms, which can be applied to any of these languages. For example, we use Stan's modeling language as an easy way to specify probability models, and our library performs inference on them.

__Computational frameworks.__
There are many computational frameworks, primarily built for deep learning: as of this date, this includes [Theano](http://deeplearning.net/software/theano/), [TensorFlow](https://www.tensorflow.org), [Torch](http://torch.ch), [Computational Graph Toolkit](http://rll.berkeley.edu/cgt/), [neon](https://github.com/NervanaSystems/neon), and [Stan Math Library](https://github.com/stan-dev/math). These are incredible back end tools, which we use rather than compete with. In terms of abstraction, our library sits at one level higher. For example, we chose to use TensorFlow among the bunch due to Python as our language of choice for machine learning research and as an investment in Google's prodigious engineering.

__High-level deep learning libraries.__
Neural network libraries such as [Keras](https://github.com/fchollet/keras) and [Lasagne](https://github.com/Lasagne/Lasagne) are at a similar abstraction level as us, but they are interested in parameterizing complicated likelihood functions in order to minimize a loss. We are more interested in inferring Bayesian hierarchical models, which can have both complicated likelihood and complicated priors (neural networks are an option but not a necessity). Therefore our goals are orthogonal, and in fact mutually benefit each other: for example, we use Keras' abstraction as a way to easily specify models parameterized by deep neural networks.

__Scalable inference.__
Implementations of black box inference algorithms are usually tied to a specific probabilistic programming language, and the majority of these algorithms focus on non-gradient based approaches using sequential Monte Carlo. We are agnostic to the modeling language. Further, we are interested in scalable approaches using gradient information from the model and optimization of an objective function. The most related in this line is [Stan](http://mc-stan.org)'s algorithms such as [ADVI](http://arxiv.org/abs/1506.03431). However, we don't aim to be as user-friendly by automating as many internals as possible. This is because this library is first and foremost a research tool, where we are interested in not abstracting away the specific mathematics but understanding the mechanics behind them. Further, our developed algorithms will eventually make its way into more mature, enterprise-level software (such as Stan). As several of us are also core developers in Stan, we do prototyping of new algorithms here, and deploy them in Stan when they're ready to be applied industry-wide.
