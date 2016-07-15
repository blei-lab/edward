Inference Algorithms
^^^^^^^^^^^^^^^^^^^^

An inference algorithm calculates the posterior for a particular model
and data set; it is the distribution of the latent variables given data,
``p(z | x)``, and is used in all downstream analyses such as prediction.
With Edward, you can develop new black box inference algorithms and also
develop custom inference algorithms which are tailored to a particular
model or restricted class of models.

There is a base ``Inference`` class, from which all inference methods
are based on. We categorize inference under two paradigms:

-  ``VariationalInference``
-  ``MonteCarlo``

(or more plainly, optimization and sampling). These inherit from
``Inference`` and each have their own default methods. See the file 
`inferences <https://github.com/blei-lab/edward/blob/master/edward/inferences.py>`__.


Consider developing a variational inference algorithm. The main method
in ``VariationalInference`` is ``run()``, which is a simple wrapper that
first runs ``initialize()`` and then in a loop runs ``update()`` and
``print_progress()``. To develop a new variational inference algorithm,
inherit from ``VariationalInference`` and write a new method for
``build_loss()``: this returns an object that TensorFlow will
automatically differentiate during optimization. The other methods have
defaults which you can update as necessary. The `inclusive KL divergence
algorithm in
`inferences <https://github.com/blei-lab/edward/blob/master/edward/inferences.py>`__
is a useful example. It writes ``build_loss()`` so that automatic
diferentiation of its return object is a tractable gradient that
minimizes KL(p\|\|q). It also modifies ``initialize()`` and
``update()``.

Consider developing a Monte Carlo algorithm. Inherit from
``MonteCarlo``. [Documentation is in progress.]

Note that you can build model-specific inference algorithms and
inference algorithms that are tailored to a smaller class than the
general class available here. There's nothing preventing you to do so,
and the general organizational paradigm and low-level functions are
still useful in such a case. You can write a class that for example
inherits from ``Inference`` directly or inherits to carry both
optimization and sampling methods.
