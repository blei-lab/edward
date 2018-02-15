"""implement the stan 8 schools example
using the recommended non-centred parameterisation

https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
http://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html
we slightly modify the stan example to:
                                         avoid improper priors
                                         avoid half-cauchy priors
make the statement of the model directly comparable

This model has a hierachy and an inferred variance - yet the example is
very simple - only the Normal distribution is used"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pystan

def main():
    # data
    J = 8
    data_y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    data_sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])    
 
    standata = dict(J=J, y=data_y, sigma=data_sigma)
    fit = pystan.stan('eight_schools.stan', data=standata, iter=100000)
    print(fit)

if __name__ == "__main__":
  main()
