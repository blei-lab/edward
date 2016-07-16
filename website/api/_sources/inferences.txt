Inference
^^^^^^^^^

An inference algorithm infers the posterior for a particular model
``p(x, z)`` and data set ``x``. It is the distribution of the latent
variables given data, ``p(z | x)``. For more details, see the
`Inference of Probability Models tutorial <>`__.

Edward uses abstract base classes and class inheritance to provide a
hierarchy of inference methods, all of which are easily extensible.
This enables fast experimentation and research on top of existing
inference methods, whether it be developing new black box inference
algorithms or developing new model-specific inference algorithms which
are tailored to a particular model or restricted class of models.
We detail this below.

\includegraphics{images/inference_structure.png}
{\small\textit{Dependency graph of inference methods.
Nodes are classes in Edward and arrows represent class inheritance.}}

There is a abstract base class ``Inference``, from which all inference
methods are derived from.

.. code:: python

  class Inference(object):
      """Base class for Edward inference methods.
      ...
      """
      def __init__(self, model, data=None):
          ...

It takes as input a probabilistic model ``model`` and dataset
``data``.
For more details, see the
`Probabilistic Models API <>`__
and
`Data API <>`__.

We categorize inference under two paradigms:
``VariationalInference`` and ``MonteCarlo`` (or more plainly,
optimization and sampling). These inherit from ``Inference`` and each
have their own default methods.

.. code:: python

  class MonteCarlo(Inference):
      """Base class for Monte Carlo inference methods.
      """
      def __init__(self, *args, **kwargs):
          super(MonteCarlo, self).__init__(*args, **kwargs)

      ...


  class VariationalInference(Inference):
      """Base class for variational inference methods.
      """
      def __init__(self, model, variational, data=None):
          """Initialization.
          ...
          """
          super(VariationalInference, self).__init__(model, data)
          self.variational = variational

      ...

Currently, Edward has most of its inference infrastructure within the
``VariationalInference`` class.
The ``MonteCarlo`` class is still under development. We welcome
contributors to make significant advances here!

Let's focus on ``VariationalInference``. In addition to a model and
data as input, ``VariationalInference`` also takes in a variational
model ``variational``, which serves as a model of the posterior
distribution. For more details, see the
`Variational Models API <>`__.

The main method in ``VariationalInference`` is ``run()``.

.. code:: python

  class VariationalInference(Inference):
      """Base class for variational inference methods.
      """
      ...
      def run(self, *args, **kwargs):
          """A simple wrapper to run variational inference.
          ...
          """
          self.initialize(*args, **kwargs)
          for t in range(self.n_iter+1):
              loss = self.update()
              self.print_progress(t, loss)

          self.finalize()

      ...

First, it calls ``initialize()`` to initialize the algorithm, such as
setting the number of iterations. Then, within a loop it calls
``update()`` which runs one step of inference, as well as
``print_progress()`` for possibly displaying diagnostics; finally, it
calls ``finalize()`` which runs the final steps as the inference
algorithm terminates.

Developing a new variational inference algorithm is as simple as
inheriting from ``VariationalInference`` or one of its derived
classes. ``VariationalInference`` implements many default methods such
as ``run()`` above. For example, ``initialize()`` creates a TensorFlow
optimizer and builds the computational graph for running the
algorithm. It calls the method ``build_loss()``, which returns a node
to differentiate for gradient-based optimization.  ``build_loss()`` is
not implemented in ``VariationalInference`` and must be defined in a
derived class defining a variational inference algorithm. As another
example, ``update()`` runs a TensorFlow session to run one step of the
optimizer. It also fetches ``self.loss`` which is a node in the
computational graph, forming the objective value given the current
state of the graph. This field must also be implemented in a derived
class.

Nothing in ``Inference`` says anything about the class of models that
an inference algorithm must work with. Thus one can build inference
algorithms which are tailored to a smaller class of models than the
general class available in Edward, or even tailor it to a single model.

Hybrid methods and novel paradigms outside of ``VariationalInference``
and ``MonteCarlo`` are also possible in Edward. For example, one can
write a class derived from ``Inference`` directly, or inherited to
carry both ``VariationalInference`` and ``MonteCarlo`` methods.

For examples of inference algorithms built in Edward, see the inference
`tutorials <>`__.
