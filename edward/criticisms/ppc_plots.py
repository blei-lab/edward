from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def ppc_density_plot(y, y_rep):
  """Create 1D kernel density plot comparing data to samples from posterior.

  Args:
    y: np.ndarray.
      A 1-D NumPy array.
    y_rep: np.ndarray.
      A 2-D NumPy array where rows represent different samples from posterior.

  Returns:
    matplotlib axes

  #### Examples

  ```python
  import matplotlib.pyplot as plt

  y = np.random.randn(20)
  y_rep = np.random.randn(20, 20)

  ed.ppc_density_plot(y, y_rep)
  plt.show()
  ```
  """
  import matplotlib.pyplot as plt
  import seaborn as sns
  ax = sns.kdeplot(y, color="maroon")

  n = y_rep.shape[0]

  for i in range(n):
    ax = sns.kdeplot(y_rep[i, :], color="maroon", alpha=0.2, linewidth=0.8)

  y_line = plt.Line2D([], [], color='maroon', label='y')
  y_rep_line = plt.Line2D([], [], color='maroon', alpha=0.2, label='y_rep')

  handles = [y_line, y_rep_line]
  labels = ['y', r'$y_{rep}$']

  ax.legend(handles, labels)

  return ax


def ppc_stat_hist_plot(y_stats, yrep_stats, stat_name=None, **kwargs):
  """Create histogram plot comparing data to samples from posterior.

  Args:
    y_stats: float.
      Float representing statistic value of observed data.
    yrep_stats: np.ndarray.
      A 1-D NumPy array.
    stat_name: string.
      Optional string value for including statistic name in legend.
    **kwargs:
      Keyword arguments used by seaborn.distplot can be given to customize plot.

  Returns:
    matplotlib axes.

  #### Examples

  ```python
  import matplotlib.pyplot as plt

  # DATA
  x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

  # MODEL
  p = Beta(1.0, 1.0)
  x = Bernoulli(probs=p, sample_shape=10)

  # INFERENCE
  qp = Beta(tf.nn.softplus(tf.Variable(tf.random_normal([]))),
            tf.nn.softplus(tf.Variable(tf.random_normal([]))))

  inference = ed.KLqp({p: qp}, data={x: x_data})
  inference.run(n_iter=500)

  # CRITICISM
  x_post = ed.copy(x, {p: qp})
  y_rep, y = ed.ppc(
      lambda xs, zs: tf.reduce_mean(tf.cast(xs[x_post], tf.float32)),
      data={x_post: x_data})

  ed.ppc_stat_hist_plot(
      y[0], y_rep, stat_name=r'$T \equiv$mean', bins=10)
  plt.show()
  ```
  """
  import matplotlib.pyplot as plt
  import seaborn as sns
  ax = sns.distplot(yrep_stats, kde=False, label=r'$T(y_{rep})$', **kwargs)

  max_value = ax.get_ylim()[1]

  plt.vlines(y_stats, ymin=0.0, ymax=max_value, label='T(y)')

  if stat_name is not None:
    plt.legend(title=stat_name)
  else:
    plt.legend()

  return ax
