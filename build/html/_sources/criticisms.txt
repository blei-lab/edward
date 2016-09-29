Criticism
---------

Criticism is about understanding the
assumptions made in one's analysis. For example, model criticism
measures the degree to which our model falsely describes the data.

Criticism techniques are simply functions which take as input data,
the probability model and variational model (binded through a latent
variable dictionary), and any additional inputs.

.. code:: python

  def criticize(data, latent_vars, ...)
    ...

Developing new criticism techniques is easy.  They can be derived from
the current techniques or built as a standalone function.

For examples of criticism techniques built in Edward, see the
criticism
`tutorials <../tutorials/>`__.
