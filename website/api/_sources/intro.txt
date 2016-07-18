Building Probabilistic Models
=============================

A probabilistic model is specified by a joint distribution `p(x,z)` of data `x` and latent variables `z`. All models in Edward are written as a class; to implement a new model, it can be written in any of the currently supported modeling languages: Stan, TensorFlow, and NumPy/SciPy.

To use Stan, simply write a Stan program in the form of a file or string. Then call it with `StanModel(file)` or `StanModel(model_code)`. Here is an example:



