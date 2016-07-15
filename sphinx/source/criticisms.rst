Criticism
^^^^^^^^^

Criticism is about explicitly understanding and analyzing the
assumptions made in one's analysis. For example, model criticism
measures the degree to which our model falsely describes the data.
For more details, see the
`Point-based evaluations tutorial <>`__
and
`Posterior predictive checks tutorial <>`__.

Criticism techniques are simply functions which take the probability
model, variational model, and data as input along with other
additional arguments. Developing new criticism techniques is easy.
They can be derived from the current techniques or built as a
standalone function.

For examples of criticism techniques built in Edward, see the
criticism
`tutorials <>`__.
