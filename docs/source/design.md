# Design Philosophy

__Edward__ serves two purposes:

1. to serve as a foundation for research;

2. to provide a unified library for inference and criticism, with modeling tools at our disposal.

As a research tool, the code base serves as a testbed for fast experimentation, so that we can easily play with extensions of current paradigms. For example, to evaluate new inference algorithms, the experiments only need to be written once and we can simply swap the inference with the newly proposed algorithm. For certain inference algorithms, this corresponds to inheriting from a current inference class and simply writing one function. This idea applies to all components of probabilistic modeling: we can leverage the built-in inference algorithms to develop new complex models, or use the built-in models and inference to develop new criticism techniques.

As an applied tool, Edward supports a wide variety of settings, ranging from classical hierarchical models on small data sets to complex deep probabilistic models on large data sets. With [TensorFlow](https://www.tensorflow.org) as a backend, Edward can leverage features such as computational graphs, distributed training, and CPU/GPU integration to deploy probabilistic modeling at scale. It also has the fundamentals: for example, linear model examples, mixture model examples, probabilistic neural networks, and stochastic and black box variational inference.

## Related Software

There are several notable themes in Edward.

__Probabilistic programming.__
There has been incredible progress recently on developing generic languages for specifying probabilistic models, or probabilistic programs. Among these include [Venture](http://probcomp.csail.mit.edu/venture/), [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/literature/index.html), [Figaro](https://www.cra.com/work/case-studies/figaro), [Stan](http://mc-stan.org), and [WebPPL](http://dippl.org/chapters/02-webppl.html). Edward focuses less on the language and more on easily enabling probabilistic modeling across all components of data analysis. For example, we use Stan's modeling language as an easy way to specify models in Edward, and our library enables fast experimentation on their inference and criticism.

__Black box inference.__
Implementations of inference algorithms are usually tied to a specific probabilistic programming language, and the majority of these are black box algorithms, using sequential Monte Carlo and making very few assumptions about the model. There are three important distinctions. First, we focus primarily on variational inference. Second, we are agnostic to the modeling language. Third, we believe inference algorithms should take advantage of as much structure as possible from the model. Thus Edward supports all types of inference algorithms, whether they be black box or model-specific, being tailored to a single model or restricted class of models. This makes Edward's probabilistic modeling a rich and practical tool for real-world applications. It also makes Edward suitable for experimenting with all types of inference.

__Computational frameworks.__
There are many computational frameworks, primarily built for deep learning: as of this date, this includes [TensorFlow](https://www.tensorflow.org),[Theano](http://deeplearning.net/software/theano/), [Torch](http://torch.ch), [Computational Graph Toolkit](http://rll.berkeley.edu/cgt/), [neon](https://github.com/NervanaSystems/neon), and [Stan Math Library](https://github.com/stan-dev/math). These are incredible tools which Edward employs as a backend. In terms of abstraction, Edward sits at one level higher. For example, we chose to use TensorFlow among the bunch due to Python as our language of choice for machine learning research and as an investment in Google's massive engineering.

__High-level deep learning libraries.__
Neural network libraries such as [Keras](https://github.com/fchollet/keras) and [Lasagne](https://github.com/Lasagne/Lasagne) are at a similar abstraction level as us, but they are primarily interested in parameterizing complicated functions for supervised learning on large datasets. We are more interested in probabilistic models which apply to a wide amount of learning tasks, and which can have both complicated likelihood and complicated priors (neural networks are an option but not a necessity). Therefore our goals are orthogonal and in fact mutually benefit each other. For example, we use Keras' abstraction as a way to easily specify models parameterized by deep neural networks.
