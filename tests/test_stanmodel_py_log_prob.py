# Compare log_prob evaluations from Stan to a NumPy/SciPy version.
import blackbox as bb
import numpy as np

from blackbox.util import PythonModel
from scipy.stats import beta, bernoulli

class BetaBernoulli(PythonModel):
    """
    p(z) = Beta(z; 1, 1)
    p(x|z) = Bernoulli(x; z)
    """
    def __init__(self, data):
        self.data = data
        self.num_vars = 1

    def _py_log_prob(self, zs):
        # This example is written for pedagogy. We recommend
        # vectorizing operations in practice.
        n_minibatch = zs.shape[0]
        lp = np.zeros(n_minibatch, dtype=np.float32)
        for b in range(n_minibatch):
            lp[b] = beta.logpdf(zs[b, :], a=1.0, b=1.0)
            for n in range(len(self.data)):
                lp[b] += bernoulli.logpmf(self.data[n], p=zs[b, :])

        return lp

data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
npmodel = BetaBernoulli(data)

model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(1.0, 1.0);
      for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
"""
data = dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

model = bb.StanModel(model_code=model_code, data=data)

print( npmodel._py_log_prob(np.array([[0.5]], dtype=np.float32)) )
print( model._py_log_prob(np.array([[0.5]], dtype=np.float32)) )
print()
print( npmodel._py_log_prob(np.array([[0.314]], dtype=np.float32)) )
print( model._py_log_prob(np.array([[0.314]], dtype=np.float32)) )
print()
print( npmodel._py_log_prob(np.array([[0.682]], dtype=np.float32)) )
print( model._py_log_prob(np.array([[0.682]], dtype=np.float32)) )
