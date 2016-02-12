import numpy as np

from scipy.stats import gaussian_kde
from models import PosteriorBernoulli, PosteriorMixturePoisson
from util import discrete_density

class VI:
  """
  Base class for variational inference methods.

  Arguments
  ----------
  niter: number of iterations
  g_num_samples: minibatch size
  g_print_samples: minibatch size for print iterations
  inner_samples: # of times to use z sample per update
  print_freq: how often to print progress
  """
  def __init__(self, niter, g_num_samples=100, g_print_samples=2000,
    inner_samples=1, print_freq=100):
    self.niter = niter
    self.g_num_samples = g_num_samples
    self.g_print_samples = g_print_samples
    self.inner_samples = inner_samples
    self.print_freq = print_freq

    # TODO fix no data for now
    self.x = 0

  def run(self):
    for i in range(self.niter):
      self._update(i)

  def _update(self, i):
    pass

  def _model_print(self):
    """Print custom things according to model."""
    if (isinstance(self.model, PosteriorBernoulli) or
        isinstance(self.model, PosteriorMixturePoisson)):
      self._discrete_print()
    else:
      pass

  def _discrete_print(self):
    pass

class HVM(VI):
  """
  Black box inference with a hierarchical variational model.
  (Ranganath et al., 2016)

  Arguments
  ----------
  model: probability model p(x, z)
  q_mf: likelihood q(z | lambda) (must be a mean-field)
  q_prior: prior q(lambda; theta)
  r_auxiliary: auxiliary r(lambda | z; phi)
  global_sample_freq: # of times to use lambda sample per update
  """
  def __init__(self, model, q_mf, q_prior, r_auxiliary,
    global_sample_freq=1, *args, **kwargs):
    VI.__init__(self, *args, **kwargs)

    self.model = model
    self.q_mf = q_mf
    self.q_prior = q_prior
    self.r_auxiliary = r_auxiliary

    self.global_sample_freq = global_sample_freq

  def _update(self, i):
    elbo = 0
    z_elbo = 0
    r_term = 0
    q_term = 0
    z_entropy = 0
    lp_sum = 0

    if i % self.print_freq == 0:
      num_samples = self.g_print_samples
    else:
      num_samples = self.g_num_samples

    for s in range(num_samples):
      if s % self.global_sample_freq == 0:
        lamda_unconst = self.q_prior.sample()
        self.r_auxiliary.reverse_samples(lamda_unconst)

      if True:
      #for i in range(self.inner_samples):
        lamda = self.q_mf.transform(lamda_unconst)
        self.q_mf.set_lamda(lamda)
        z = self.q_mf.sample()

        elbo_in = 0
        elbo_in = self.model.log_prob(self.x, z)
        lp_sum += self.model.log_prob(self.x, z) / num_samples
        for d in range(self.model.num_vars):
          elbo_in -= self.q_mf.log_prob_zi(d, z)

        z_elbo += elbo_in
        for d in range(self.model.num_vars):
          z_entropy += self.q_mf.log_prob_zi(d, z)

        z_entropy = z_entropy / num_samples

        reward_signal = self._reward_signal(z)
        self.q_prior.add_grad(lamda_unconst, reward_signal)

        q_term -= self.q_prior.log_prob(lamda_unconst) / num_samples
        elbo_in -= self.q_prior.log_prob(lamda_unconst)
        elbo_in += self.r_auxiliary.log_prob(z)
        r_term += self.r_auxiliary.log_prob(z) / num_samples

    elbo += elbo_in / num_samples

    self.q_prior.normalize_grad(num_samples * self.inner_samples)
    self.r_auxiliary.normalize_grad(num_samples * self.inner_samples)

    eta = 1e-3
    # TODO the iteration count shouldn't include print iterations
    #eta = (i + 1) ** -.7
    if not (i % self.print_freq == 0):
      self.q_prior.update(eta)
      self.r_auxiliary.update(eta)

    if (i % self.print_freq == 0):
      print "ITER: %d:" % i
      print ("Elbo, Z_elbo, r_term, q_term, z_entropy, log_p: %f, %f, %f, %f, %f, %f" %
             (elbo,
              z_elbo/num_samples, (r_term), q_term, z_entropy, lp_sum))
      self._model_print()

  def _reward_signal(self, z):
    # reward signal, excluding the term
    # -1 * grad_{lambda} log q(lambda; theta)
    reward_signal = np.zeros(self.q_mf.num_params)
    num_params_per_var = self.q_mf.num_params / self.q_mf.num_vars
    for d in range(self.q_mf.num_vars):
      idx = [i for i in range(num_params_per_var*d, \
                              num_params_per_var*(d + 1))]
      score_zi = self.q_mf.score_zi(d, z)
      q_mf_grad = score_zi * (self.model.log_prob(self.x, z) - \
                               self.q_mf.log_prob_zi(d, z))
      reward_signal[idx] += q_mf_grad
      reward_signal[idx] += score_zi * self.r_auxiliary.likelihood(d, z[d])

    # r's gradient is updated here
    reward_signal += self.r_auxiliary.add_grad(z)
    return reward_signal

  def _discrete_print(self):
    num_test_samples = 5000
    z_samples = np.zeros((num_test_samples, self.model.num_vars))
    for r in range(num_test_samples):
      lamda_unconst = self.q_prior.sample()
      lamda = self.q_mf.transform(lamda_unconst)
      self.q_mf.set_lamda(lamda)
      z_samples[r, :] = self.q_mf.sample()

    q_z = discrete_density(z_samples)
    kl_estimate = 0
    for r in range(num_test_samples):
      try:
        z_sample = z_samples[r, :]
        elem = q_z
        for d in range(self.model.num_vars):
          elem = elem[z_sample[d]]
      except:
        # TODO
        # This shouldn't ever happen but for some reason it does.
        print "Error computing KL; skipping..."
        return

      kl_estimate += np.log(elem) - self.model.log_prob(self.x, z_sample)

    if isinstance(self.model, PosteriorBernoulli):
      print "Variational approximation:"
      print q_z
    print "Variational prior parameters:"
    self.q_prior.print_params()
    print "KL: "
    print kl_estimate / num_test_samples

  # TODO
  #def _continuous_print(self):
  #  num_test_samples = 5000
  #  z_samples = np.zeros((self.model.num_vars, num_test_samples))
  #  for r in range(num_test_samples):
  #    lamda_unconst = self.q_prior.sample()
  #    lamda = self.q_mf.transform(lamda_unconst)
  #    self.q_mf.set_lamda(lamda)
  #    z_samples[:, r] = self.q_mf.sample()

  #  q_z = gaussian_kde(z_samples)
  #  kl_estimate = 0
  #  for r in range(num_test_samples):
  #      z_sample = z_samples[:, r]
  #      kl_estimate += q_z.logpdf(z_sample) - self.model.log_prob(self.x, z_sample)

  #  print "KL: "
  #  print kl_estimate / num_test_samples

class MFVI(VI):
  """
  Mean-field (black box) variational inference.
  (Ranganath et al., 2014)
  TODO maybe this stuff should go under the base class

  Arguments
  ----------
  model: probability model p(x, z)
  q_mf: likelihood q(z | lambda) (must be a mean-field)
  """
  def __init__(self, model, q_mf, *args, **kwargs):
    VI.__init__(self, *args, **kwargs)
    self.model = model
    self.q_mf = q_mf

    # TODO move these inside the q_mf class
    c = 1e-4
    self.q_mf_grad = np.zeros(self.q_mf.num_params)
    self.q_mf_grad_sum = c * np.ones(self.q_mf.num_params)
    self.q_mf_momentum = np.zeros(self.q_mf.num_params)

  def _update(self, i):
    elbo = 0

    if i % self.print_freq == 0:
      num_samples = self.g_print_samples
    else:
      num_samples = self.g_num_samples

    for s in range(num_samples):
      if True:
      #for i in range(self.inner_samples):
        z = self.q_mf.sample()

        # TODO this doesn't need to be calculated for non-print
        # iterations
        elbo_in = 0
        elbo_in = self.model.log_prob(self.x, z)
        for d in range(self.model.num_vars):
            elbo_in -= self.q_mf.log_prob_zi(d, z)

        reward_signal = self._reward_signal(z)
        self.q_mf_add_grad(reward_signal)

    self.q_mf_normalize_grad(num_samples * self.inner_samples)
    elbo += elbo_in / num_samples

    eta = 1e-3
    if not (i % self.print_freq == 0):
      self.q_mf_update(eta)

    if (i % self.print_freq == 0):
      print "ITER: %d:" % i
      print "Elbo: %f" % elbo
      self._model_print()

  def _reward_signal(self, z):
    reward_signal = np.zeros(self.q_mf.num_params)
    num_params_per_var = self.q_mf.num_params / self.q_mf.num_vars
    for d in range(self.q_mf.num_vars):
      idx = [i for i in range(num_params_per_var*d, \
                              num_params_per_var*(d + 1))]
      reward_signal[idx] += self.q_mf.score_zi(d, z) * ( \
        self.model.log_prob(self.x, z) - \
        self.q_mf.log_prob_zi(d, z))

    return reward_signal

  def _discrete_print(self):
    num_test_samples = 5000
    z_samples = np.zeros((num_test_samples, self.model.num_vars))
    for r in range(num_test_samples):
      z_samples[r, :] = self.q_mf.sample()

    q_z = discrete_density(z_samples)
    kl_estimate = 0
    for r in range(num_test_samples):
      try:
        z_sample = z_samples[r, :]
        elem = q_z
        for d in range(self.model.num_vars):
          elem = elem[z_sample[d]]
      except:
        # TODO
        # This shouldn't ever happen but for some reason it does.
        print "Error computing KL; skipping..."
        return

      kl_estimate += np.log(elem) - self.model.log_prob(self.x, z_sample)

    if isinstance(self.model, PosteriorBernoulli):
      print "Variational approximation:"
      print q_z
    print "Variational likelihood parameters:"
    # TODO hardcoded should have a print_params() method
    #print self.q_mf.lamda
    print "KL: "
    print kl_estimate / num_test_samples

  def q_mf_add_grad(self, vec_grad):
    self.q_mf_grad += vec_grad

  def q_mf_normalize_grad(self, normalization):
    self.q_mf_grad /= normalization

  def q_mf_update(self, eta):
    self.q_mf_grad_sum = (1 - .1) * self.q_mf_grad_sum + .1 * self.q_mf_grad * self.q_mf_grad

    alpha = 0.9
    self.q_mf_momentum = alpha * self.q_mf_momentum + eta * self.q_mf_grad / np.sqrt(self.q_mf_grad_sum)

    self.q_mf.add_lamda(self.q_mf_momentum)

    self.q_mf_grad *= 0

class AlphaVI(MFVI):
  """
  alpha-divergence

  Arguments
  ----------
  model: probability model p(x, z)
  q_mf: likelihood q(z | lambda) (must be a mean-field)
  """
  def __init__(self, alpha, *args, **kwargs):
    MFVI.__init__(self, *args, **kwargs)
    self.alpha = alpha

  def _update(self, i):
    elbo = 0

    if i % self.print_freq == 0:
      num_samples = self.g_print_samples
    else:
      num_samples = self.g_num_samples

    for s in range(num_samples):
      if True:
      #for i in range(self.inner_samples):
        z = self.q_mf.sample()

        # TODO this doesn't need to be calculated for non-print
        # iterations
        elbo_in = 0
        elbo_in = self.model.log_prob(self.x, z)
        for d in range(self.model.num_vars):
            elbo_in -= self.q_mf.log_prob_zi(d, z)

        elbo_in = np.exp(elbo_in) ** (1 - self.alpha)

        reward_signal = self._reward_signal(z)
        self.q_mf_add_grad(reward_signal)

    self.q_mf_normalize_grad(num_samples * self.inner_samples)
    elbo += elbo_in / num_samples

    eta = 1e-3
    if not (i % self.print_freq == 0):
      self.q_mf_update(eta)

    if (i % self.print_freq == 0):
      print "ITER: %d:" % i
      print "alpha-Elbo: %f" % elbo
      self._model_print()

  def _reward_signal(self, z):
    reward_signal = np.zeros(self.q_mf.num_params)
    num_params_per_var = self.q_mf.num_params / self.q_mf.num_vars

    q_mf_log_prob = 0
    for d in range(self.q_mf.num_vars):
      idx = [i for i in range(num_params_per_var*d, \
                              num_params_per_var*(d + 1))]
      q_mf_log_prob += self.q_mf.log_prob_zi(d, z)

    temp = self.model.log_prob(self.x, z) - q_mf_log_prob
    temp = np.exp(temp) ** (1 - self.alpha)
    #temp = (1 - self.alpha) * temp # biased version
    for d in range(self.q_mf.num_vars):
      idx = [i for i in range(num_params_per_var*d, \
                              num_params_per_var*(d + 1))]
      reward_signal[idx] += self.q_mf.score_zi(d, z) * temp

    return reward_signal
