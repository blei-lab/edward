import edward as ed
import numpy as np
from edward import PyMC3Model, Variational, Beta
import pymc3 as pm
import theano
import numpy as np

data_shared = theano.shared(np.zeros(1))

with pm.Model() as model:
    beta = pm.Beta('beta', 1, 1, transform=None)
    out = pm.Bernoulli('data',
                       beta,
                       observed=data_shared)

data = ed.Data(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]))
m = PyMC3Model(model, data_shared)
variational = Variational()
variational.add(Beta(m.num_vars))
inference = ed.MFVI(m, variational, data)
inference.run(n_iter=10000)
